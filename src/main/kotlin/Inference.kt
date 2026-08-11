/*
 * Inference.kt — KV-cache generation.
 *
 * The training forward pass (Model.kt) reprocesses the WHOLE sequence every step
 * and builds an autograd graph. For generation we don't need gradients, and we
 * don't need to recompute the past: attention's keys (K) and values (V) for
 * earlier tokens never change, so we cache them.
 *
 * With the cache, each new token only:
 *   1. computes its own q, k, v,
 *   2. appends k, v to the per-layer cache,
 *   3. attends its q over all cached k/v.
 * That turns per-step work from O(t · d) matmuls over t past tokens into O(t)
 * dot-products — the standard trick that makes LLM decoding fast.
 *
 * This is a plain-array reimplementation of the same math the Tensor graph does
 * (no autograd), which also makes the inference path explicit and easy to read.
 * `kvCacheMatchesFullForward` in the tests asserts it matches Model.forward.
 */

import java.util.Random

class KVGenerator(private val model: GPT) {
    private val cfg = model.cfg
    private val d = cfg.nEmbed
    private val H = cfg.nHead
    private val hd = d / H
    private val base = 10000.0

    private val kCache = Array(cfg.nLayer) { ArrayList<DoubleArray>() }
    private val vCache = Array(cfg.nLayer) { ArrayList<DoubleArray>() }

    /** Number of tokens processed so far (also the next token's position). */
    var length = 0; private set

    fun reset() {
        for (l in 0 until cfg.nLayer) { kCache[l].clear(); vCache[l].clear() }
        length = 0
    }

    /** Feed one token; returns the logits (over the vocab) for the NEXT token. */
    fun step(tokenId: Int): DoubleArray {
        val pos = length
        val x = DoubleArray(d)
        val te = model.tokEmb.data
        for (j in 0 until d) x[j] = te[tokenId * d + j]
        model.posEmb?.let { pe -> for (j in 0 until d) x[j] += pe.data[pos * d + j] }

        for (l in 0 until cfg.nLayer) {
            val b = model.blocks[l]
            val a = attention(l, layerNorm(x, b.ln1g.data, b.ln1b.data), pos)
            for (j in 0 until d) x[j] += a[j]                         // residual
            val m = mlp(b.mlp, layerNorm(x, b.ln2g.data, b.ln2b.data))
            for (j in 0 until d) x[j] += m[j]                         // residual
        }
        val xf = layerNorm(x, model.lnFg.data, model.lnFb.data)
        length++
        return matVec(xf, model.head.data, d, cfg.vocabSize)
    }

    private fun attention(l: Int, x: DoubleArray, pos: Int): DoubleArray {
        val att = model.blocks[l].attn
        val q = matVec(x, att.wq.data, d, d)
        val k = matVec(x, att.wk.data, d, d)
        val v = matVec(x, att.wv.data, d, d)
        if (cfg.useRope) { rope(q, pos); rope(k, pos) }   // k stored already rotated
        kCache[l].add(k); vCache[l].add(v)

        val t = kCache[l].size
        val out = DoubleArray(d)
        val scale = 1.0 / Math.sqrt(hd.toDouble())
        val scores = DoubleArray(t)
        for (h in 0 until H) {
            val off = h * hd
            var mx = Double.NEGATIVE_INFINITY
            for (ti in 0 until t) {
                val kt = kCache[l][ti]
                var s = 0.0
                for (j in 0 until hd) s += q[off + j] * kt[off + j]
                s *= scale
                scores[ti] = s
                if (s > mx) mx = s
            }
            var sum = 0.0
            for (ti in 0 until t) { val e = Math.exp(scores[ti] - mx); scores[ti] = e; sum += e }
            for (ti in 0 until t) {
                val w = scores[ti] / sum
                val vt = vCache[l][ti]
                for (j in 0 until hd) out[off + j] += w * vt[off + j]
            }
        }
        return matVec(out, att.wo.data, d, d)
    }

    private fun mlp(m: MLP, x: DoubleArray): DoubleArray {
        val hidden = 4 * d
        val h = matVec(x, m.w1.data, d, hidden)
        for (o in 0 until hidden) h[o] = gelu(h[o] + m.b1.data[o])
        val out = matVec(h, m.w2.data, hidden, d)
        for (o in 0 until d) out[o] += m.b2.data[o]
        return out
    }

    private fun rope(vec: DoubleArray, pos: Int) {
        val half = hd / 2
        for (h in 0 until H) {
            val off = h * hd
            for (kk in 0 until half) {
                val ang = pos * Math.pow(base, -2.0 * kk / hd)
                val c = Math.cos(ang); val s = Math.sin(ang)
                val a = vec[off + 2 * kk]; val b = vec[off + 2 * kk + 1]
                vec[off + 2 * kk] = a * c - b * s
                vec[off + 2 * kk + 1] = a * s + b * c
            }
        }
    }

    private fun layerNorm(x: DoubleArray, gamma: DoubleArray, beta: DoubleArray): DoubleArray {
        val n = x.size
        var mean = 0.0; for (v in x) mean += v; mean /= n
        var varc = 0.0; for (v in x) { val e = v - mean; varc += e * e }; varc /= n
        val inv = 1.0 / Math.sqrt(varc + 1e-5)
        val out = DoubleArray(n)
        for (j in 0 until n) out[j] = gamma[j] * ((x[j] - mean) * inv) + beta[j]
        return out
    }

    private fun gelu(x: Double): Double {
        val c = Math.sqrt(2.0 / Math.PI)
        return 0.5 * x * (1.0 + Math.tanh(c * (x + 0.044715 * x * x * x)))
    }

    /** y = xᵀ · W, where W is (inN x outN) row-major. */
    private fun matVec(x: DoubleArray, w: DoubleArray, inN: Int, outN: Int): DoubleArray {
        val y = DoubleArray(outN)
        for (i in 0 until inN) {
            val xi = x[i]
            if (xi == 0.0) continue
            val row = i * outN
            for (o in 0 until outN) y[o] += xi * w[row + o]
        }
        return y
    }
}

/**
 * Generate with the KV-cache. Because positions must stay within the model's
 * learned range, total length (prompt + generated) is capped at blockSize — the
 * model can't attend beyond its context window anyway. For longer, sliding-window
 * output use `generate` (Train.kt).
 */
fun generateKV(model: GPT, tok: Tokenizer, prompt: String, maxNew: Int, sampler: Sampler, rng: Random): String {
    val cap = model.cfg.blockSize
    var ids = tok.encode(prompt.ifEmpty { " " })
    if (ids.size > cap) ids = ids.copyOfRange(ids.size - cap, ids.size)

    val gen = KVGenerator(model)
    val out = ids.toMutableList()
    var logits = DoubleArray(model.cfg.vocabSize)
    for (id in ids) logits = gen.step(id)

    var produced = 0
    while (produced < maxNew && gen.length < cap) {
        val next = sampleFrom(logits, sampler, rng)
        out.add(next); produced++
        if (gen.length >= cap) break
        logits = gen.step(next)
    }
    return tok.decode(out.toIntArray())
}
