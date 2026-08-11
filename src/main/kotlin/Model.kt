/*
 * Model.kt — a minimal GPT (decoder-only transformer).
 *
 * Architecture (same shape as GPT-2, just tiny):
 *
 *   tokens ──► token embedding  ┐
 *                               ├─(add)─► x ─►[ Block ]x N ─► final LayerNorm ─► logits
 *   positions ► position embed  ┘
 *
 *   Each Block: x = x + Attention(LayerNorm(x))     // "communicate" across time
 *               x = x + MLP(LayerNorm(x))           // "think" per position
 *
 * Attention is the only place information moves *between* tokens. Everything else
 * acts on each position independently. The causal mask makes it a language model:
 * position i can only look at positions 0..i, so predicting the next token never
 * peeks at the answer.
 */

import java.util.Random

/** Hyperparameters. Keep tiny so it trains on a CPU in seconds. */
data class Config(
    val vocabSize: Int,
    val blockSize: Int = 16,   // max context length (tokens the model can see)
    val nEmbed: Int = 64,      // embedding / hidden dimension
    val nHead: Int = 4,        // number of attention heads
    val nLayer: Int = 2,       // number of transformer blocks
    val useRope: Boolean = false, // rotary positions instead of a learned table
    val tieWeights: Boolean = false, // share tokEmb as the output projection
    val dropout: Double = 0.0, // dropout probability (training only)
)

/** Carries the dropout probability and RNG through a training forward pass. */
class DropCtx(val p: Double, val rng: Random)

/** Apply dropout when a context is present (training); otherwise pass through. */
fun DropCtx?.maybeDropout(x: Tensor): Tensor =
    if (this == null) x else Tensor.dropout(x, p, training = true, rng = rng)

/** A single causal multi-head self-attention layer. */
class Attention(cfg: Config, rng: Random) {
    val nHead = cfg.nHead
    val headDim = cfg.nEmbed / cfg.nHead
    val useRope = cfg.useRope
    val wq = Tensor.param(cfg.nEmbed, cfg.nEmbed, rng)
    val wk = Tensor.param(cfg.nEmbed, cfg.nEmbed, rng)
    val wv = Tensor.param(cfg.nEmbed, cfg.nEmbed, rng)
    val wo = Tensor.param(cfg.nEmbed, cfg.nEmbed, rng)

    fun params() = listOf(wq, wk, wv, wo)

    fun forward(x: Tensor, sink: AttnSink? = null, layer: Int = -1): Tensor {
        val q = x matmul wq   // (T x d): "what am I looking for?"
        val k = x matmul wk   // (T x d): "what do I offer?"
        val v = x matmul wv   // (T x d): "what will I pass on?"
        val scale = 1.0 / Math.sqrt(headDim.toDouble())
        val heads = ArrayList<Tensor>(nHead)
        for (h in 0 until nHead) {
            val c0 = h * headDim; val c1 = c0 + headDim
            var qh = q.sliceCols(c0, c1)
            var kh = k.sliceCols(c0, c1)
            val vh = v.sliceCols(c0, c1)
            // RoPE injects position by rotating queries and keys (not values).
            if (useRope) { qh = Tensor.rope(qh); kh = Tensor.rope(kh) }
            // Affinity of every token to every other token, then mask the future.
            var scores = (qh matmul kh.transpose()).scale(scale)   // (T x T)
            scores = Tensor.causalMask(scores)
            val attn = scores.softmaxRows()                        // rows sum to 1
            sink?.record(layer, h, attn)                           // for the visualizer
            heads.add(attn matmul vh)                              // weighted sum of values
        }
        val concat = Tensor.concatCols(heads)  // (T x d)
        return concat matmul wo                 // mix information across heads
    }
}

/** Optional collector for attention matrices during a forward pass. */
class AttnSink {
    /** One entry per (layer, head): a T x T matrix of attention weights. */
    val records = ArrayList<Triple<Int, Int, Array<DoubleArray>>>()
    fun record(layer: Int, head: Int, attn: Tensor) {
        val t = attn.rows
        val m = Array(t) { i -> DoubleArray(t) { j -> attn.at(i, j) } }
        records.add(Triple(layer, head, m))
    }
}

/** Position-wise feed-forward network: d -> 4d -> gelu -> d. */
class MLP(cfg: Config, rng: Random) {
    val w1 = Tensor.param(cfg.nEmbed, 4 * cfg.nEmbed, rng)
    val b1 = Tensor.zeros(1, 4 * cfg.nEmbed)
    val w2 = Tensor.param(4 * cfg.nEmbed, cfg.nEmbed, rng)
    val b2 = Tensor.zeros(1, cfg.nEmbed)

    fun params() = listOf(w1, b1, w2, b2)

    fun forward(x: Tensor): Tensor {
        val h = ((x matmul w1) + b1).gelu()
        return (h matmul w2) + b2
    }
}

/** One transformer block: attention then MLP, each with a residual connection. */
class Block(cfg: Config, rng: Random) {
    val ln1g = ones(cfg.nEmbed); val ln1b = Tensor.zeros(1, cfg.nEmbed)
    val ln2g = ones(cfg.nEmbed); val ln2b = Tensor.zeros(1, cfg.nEmbed)
    val attn = Attention(cfg, rng)
    val mlp = MLP(cfg, rng)

    fun params() = listOf(ln1g, ln1b, ln2g, ln2b) + attn.params() + mlp.params()

    fun forward(x0: Tensor, sink: AttnSink? = null, layer: Int = -1, drop: DropCtx? = null): Tensor {
        // Residual ("+"): the block learns a *correction* to x, which keeps
        // gradients healthy in deep stacks. Dropout is applied to each sub-layer's
        // output before it's added back (residual dropout), training only.
        val a = attn.forward(Tensor.layerNorm(x0, ln1g, ln1b), sink, layer)
        val x1 = x0 + drop.maybeDropout(a)
        val m = mlp.forward(Tensor.layerNorm(x1, ln2g, ln2b))
        return x1 + drop.maybeDropout(m)
    }

    companion object {
        fun ones(d: Int): Tensor { val t = Tensor.zeros(1, d); t.data.fill(1.0); return t }
    }
}

/** The full model. */
class GPT(val cfg: Config, seed: Long = 1234L) {
    private val rng = Random(seed)
    val tokEmb = Tensor.param(cfg.vocabSize, cfg.nEmbed, rng)   // one vector per token id
    // Learned absolute positions live in a table; RoPE needs none (it rotates
    // Q/K inside attention instead), so we don't allocate one.
    val posEmb: Tensor? = if (cfg.useRope) null else Tensor.param(cfg.blockSize, cfg.nEmbed, rng)
    val blocks = List(cfg.nLayer) { Block(cfg, rng) }
    val lnFg = Block.ones(cfg.nEmbed); val lnFb = Tensor.zeros(1, cfg.nEmbed)
    // Weight tying: reuse tokEmb (vocab x d) transposed as the output projection
    // instead of a separate head (d x vocab). Saves vocab*d params and couples the
    // "meaning" of a token in and out. When not tied, a dedicated head is learned.
    val head: Tensor? = if (cfg.tieWeights) null else Tensor.param(cfg.nEmbed, cfg.vocabSize, rng)

    private val dropRng = Random(seed + 777)

    fun parameters(): List<Tensor> =
        listOfNotNull(tokEmb, posEmb, lnFg, lnFb, head) + blocks.flatMap { it.params() }

    /** Forward pass for one sequence of token ids -> logits (T x vocab). */
    fun forward(idx: IntArray, sink: AttnSink? = null, train: Boolean = false): Tensor {
        val drop = if (train && cfg.dropout > 0.0) DropCtx(cfg.dropout, dropRng) else null
        val t = idx.size
        var x = Tensor.gatherRows(tokEmb, idx)
        if (posEmb != null) x = x + Tensor.gatherRows(posEmb, IntArray(t) { it })
        x = drop.maybeDropout(x)                                   // embedding dropout
        for ((layer, b) in blocks.withIndex()) x = b.forward(x, sink, layer, drop)
        x = Tensor.layerNorm(x, lnFg, lnFb)
        return if (head != null) x matmul head else x matmul tokEmb.transpose()
    }

    /** Forward + cross-entropy loss against the next-token targets (train mode). */
    fun loss(idx: IntArray, targets: IntArray): Tensor =
        Tensor.crossEntropy(forward(idx, train = true), targets)
}
