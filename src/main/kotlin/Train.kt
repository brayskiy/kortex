/*
 * Train.kt — optimizer, gradient check, training loop, sampling, and CLI.
 *
 * Run modes (first CLI arg):
 *   gradcheck     -> verify backprop matches numerical gradients (proves the math)
 *   train [char|bpe] -> train on the built-in corpus, then generate (default: char)
 *   attn  [char|bpe] -> train briefly, then visualize attention weights
 *   (no arg)      -> gradcheck + train(char)
 */

import java.util.Random
import kotlin.math.sqrt

/** The built-in toy corpus: a few famous sentences on one line. */
fun corpus(): String = """
    to be or not to be that is the question.
    all that glitters is not gold.
    the quick brown fox jumps over the lazy dog.
    knowledge is power and power is knowledge.
""".trimIndent().replace("\n", " ").replace(Regex(" +"), " ")

/** Adam optimizer with one moment-pair of state per parameter tensor. */
class Adam(private val params: List<Tensor>, private val lr: Double = 3e-3) {
    private val m = params.map { DoubleArray(it.data.size) }
    private val v = params.map { DoubleArray(it.data.size) }
    private var t = 0
    private val b1 = 0.9; private val b2 = 0.999; private val eps = 1e-8

    fun zeroGrad() = params.forEach { it.grad.fill(0.0) }

    fun step() {
        t++
        for (p in params.indices) {
            val prm = params[p]; val mp = m[p]; val vp = v[p]
            for (i in prm.data.indices) {
                val g = prm.grad[i]
                mp[i] = b1 * mp[i] + (1 - b1) * g
                vp[i] = b2 * vp[i] + (1 - b2) * g * g
                val mHat = mp[i] / (1 - Math.pow(b1, t.toDouble()))
                val vHat = vp[i] / (1 - Math.pow(b2, t.toDouble()))
                prm.data[i] -= lr * mHat / (sqrt(vHat) + eps)
            }
        }
    }
}

/** Draw one token id from a probability distribution (with temperature). */
fun sample(logits: DoubleArray, temperature: Double, rng: Random): Int {
    val n = logits.size
    val p = DoubleArray(n)
    var mx = Double.NEGATIVE_INFINITY
    for (x in logits) mx = maxOf(mx, x)
    var sum = 0.0
    for (i in 0 until n) { val e = Math.exp((logits[i] - mx) / temperature); p[i] = e; sum += e }
    var r = rng.nextDouble() * sum
    for (i in 0 until n) { r -= p[i]; if (r <= 0) return i }
    return n - 1
}

/** Autoregressive generation: feed the model its own output, one token at a time. */
fun generate(model: GPT, tok: Tokenizer, prompt: String, maxNew: Int, temperature: Double, rng: Random): String {
    var ids = tok.encode(prompt.ifEmpty { " " })
    for (step in 0 until maxNew) {
        val context = if (ids.size <= model.cfg.blockSize) ids else ids.copyOfRange(ids.size - model.cfg.blockSize, ids.size)
        val logits = model.forward(context)
        val last = DoubleArray(model.cfg.vocabSize) { logits.at(context.size - 1, it) }
        val next = sample(last, temperature, rng)
        ids = ids + next
    }
    return tok.decode(ids)
}

/**
 * Numerical gradient check: compare analytic gradients (from backward()) against
 * finite differences (loss(w+eps) - loss(w-eps)) / 2eps. If these agree, the
 * autograd engine and every op's backward are correct. This is how you *trust*
 * a hand-written neural net.
 */
/**
 * Compute the max relative error between analytic gradients (backward()) and
 * finite differences over a spot-check of parameters. Small error => backprop
 * is correct. Pure/deterministic so tests can assert on it.
 */
fun computeMaxGradError(): Double {
    val text = "hello world, transformers!"
    val tok = CharTokenizer(text)
    val cfg = Config(vocabSize = tok.vocabSize, blockSize = 8, nEmbed = 16, nHead = 2, nLayer = 2)
    val model = GPT(cfg, seed = 7)
    val idx = tok.encode("hello wo")
    val tgt = tok.encode("ello wor")

    model.parameters().forEach { it.grad.fill(0.0) }
    model.loss(idx, tgt).backward()

    val eps = 1e-5
    val rng = Random(0)
    var maxRel = 0.0
    // Spot-check a handful of random parameters across a few tensors.
    val toCheck = listOf(model.head, model.tokEmb, model.blocks[0].attn.wq, model.blocks[1].mlp.w1, model.lnFg)
    for (p in toCheck) {
        repeat(6) {
            val i = rng.nextInt(p.data.size)
            val analytic = p.grad[i]
            val orig = p.data[i]
            p.data[i] = orig + eps; val lp = model.loss(idx, tgt).data[0]
            p.data[i] = orig - eps; val lm = model.loss(idx, tgt).data[0]
            p.data[i] = orig
            val numeric = (lp - lm) / (2 * eps)
            val rel = Math.abs(analytic - numeric) / (Math.abs(analytic) + Math.abs(numeric) + 1e-9)
            maxRel = maxOf(maxRel, rel)
        }
    }
    return maxRel
}

fun gradCheck() {
    val maxRel = computeMaxGradError()
    println("Gradient check: max relative error = %.2e".format(maxRel))
    println(if (maxRel < 1e-4) "PASS — backprop is correct.\n" else "FAIL — check the op backwards.\n")
}

/** Build a tokenizer of the requested kind over the corpus. */
fun makeTokenizer(kind: String, text: String): Tokenizer = when (kind) {
    "bpe" -> BpeTokenizer.train(text, targetVocab = 300)
    else -> CharTokenizer(text)
}

/** Train a model on `text` with `tok`; returns the trained model. */
fun trainModel(
    tok: Tokenizer, text: String,
    steps: Int, batch: Int = 12, lr: Double = 3e-3,
    cfg: Config, seed: Long = 1234L, verbose: Boolean = true,
): GPT {
    val data = tok.encode(text)
    val model = GPT(cfg, seed)
    val opt = Adam(model.parameters(), lr)
    val rng = Random(42)
    require(data.size > cfg.blockSize + 1) { "corpus too short for blockSize=${cfg.blockSize}" }

    if (verbose) println(
        "chars=${text.length}  tokens=${data.size}  vocab=${tok.vocabSize}  " +
            "params=${model.parameters().sumOf { it.data.size }}"
    )

    var running = 0.0
    for (step in 1..steps) {
        var lossVal = 0.0
        opt.zeroGrad()
        for (b in 0 until batch) {
            val start = rng.nextInt(data.size - cfg.blockSize - 1)
            val idx = data.copyOfRange(start, start + cfg.blockSize)
            val tgt = data.copyOfRange(start + 1, start + cfg.blockSize + 1)
            val l = model.loss(idx, tgt)
            l.backward()               // gradients accumulate across the batch
            lossVal += l.data[0]
        }
        model.parameters().forEach { p -> for (i in p.grad.indices) p.grad[i] /= batch }
        opt.step()

        running += lossVal / batch
        if (verbose && step % 250 == 0) {
            println("step %4d  loss %.4f".format(step, running / 250))
            running = 0.0
        }
    }
    return model
}

fun train(kind: String) {
    val text = corpus()
    val tok = makeTokenizer(kind, text)
    val blockSize = if (kind == "bpe") 16 else 24   // BPE tokens cover more text per step
    val cfg = Config(vocabSize = tok.vocabSize, blockSize = blockSize, nEmbed = 64, nHead = 4, nLayer = 2)

    println("tokenizer=$kind")
    val model = trainModel(tok, text, steps = 1500, cfg = cfg)

    val rng = Random(7)
    println("\n--- samples (temperature 0.8) ---")
    for (prompt in listOf("to be", "the ", "knowledge")) {
        println("[$prompt] -> ${generate(model, tok, prompt, maxNew = 60, temperature = 0.8, rng = rng)}")
    }
}

/** Train briefly, then render attention heatmaps for a sample string. */
fun attnMode(kind: String) {
    val text = corpus()
    val tok = makeTokenizer(kind, text)
    val blockSize = if (kind == "bpe") 16 else 24
    val cfg = Config(vocabSize = tok.vocabSize, blockSize = blockSize, nEmbed = 48, nHead = 3, nLayer = 2)

    println("tokenizer=$kind — training a small model before visualizing...")
    val model = trainModel(tok, text, steps = 800, cfg = cfg)

    val probe = "the quick brown".take(blockSize)
    visualizeAttention(model, tok, probe, htmlPath = "attention.html")
}

fun main(args: Array<String>) {
    val mode = args.getOrNull(0) ?: "all"
    val kind = args.getOrNull(1) ?: "char"
    when (mode) {
        "gradcheck" -> gradCheck()
        "train" -> train(kind)
        "attn" -> attnMode(kind)
        else -> { gradCheck(); train("char") }
    }
}
