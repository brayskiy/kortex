/*
 * Train.kt — optimizer, gradient check, training loop, sampling, and CLI.
 *
 * Run modes (first CLI arg):
 *   gradcheck     -> verify backprop matches numerical gradients (proves the math)
 *   train [char|bpe] -> train on the built-in corpus, then generate (default: char)
 *   attn  [char|bpe] -> train briefly, then visualize attention weights
 *   (no arg)      -> gradcheck + train(char)
 */

import java.io.File
import java.util.Random
import kotlin.math.sqrt

/** The built-in toy corpus: a few famous sentences on one line. */
fun corpus(): String = """
    to be or not to be that is the question.
    all that glitters is not gold.
    the quick brown fox jumps over the lazy dog.
    knowledge is power and power is knowledge.
""".trimIndent().replace("\n", " ").replace(Regex(" +"), " ")

/**
 * A larger built-in corpus (public-domain Aesop's fables, ~2 KB) — big enough to
 * hold out a validation set and watch dropout/tying affect generalization,
 * without needing an external file. Use via `train --sample`.
 */
fun sampleCorpus(): String = """
    a hungry fox saw some clusters of ripe black grapes hanging from a trellised vine.
    she resorted to all her tricks to get at them, but wearied herself in vain, for
    she could not reach them. at last she turned away, hiding her disappointment and
    saying: the grapes are sour, and not ripe as i thought.

    a crow was sitting on a branch of a tree with a piece of cheese in her beak when a
    fox observed her and set his wits to work to discover some way of getting the cheese.
    coming and standing under the tree he looked up and said, what a noble bird i see
    above me. her beauty is without equal, and if only her voice is as sweet as her
    looks she ought to be the queen of the birds. the crow was hugely flattered, and
    just to show that she could sing she gave a loud caw. down came the cheese, and the
    fox snatching it up said, you have a voice, but what you want is wits.

    a dog was crossing a plank bridge over a stream with a piece of meat in his mouth
    when he saw his own reflection in the water. he thought it was another dog with a
    piece of meat twice as big, so he let go his own to snatch the larger, and lost both.

    a wolf came upon a lamb straying from the flock and wished to find some pretext for
    devouring her. you are the one who insulted me a year ago, said the wolf. indeed,
    said the lamb, i was not born then. no matter, replied the wolf, one excuse is as
    good as another, and he ate her all the same. the tyrant will always find a pretext
    for his tyranny, and it is useless for the innocent to try to escape by reasoning.

    a town mouse once visited a country mouse who gave him beans and bacon and bread.
    the town mouse laughed and said, my poor friend, you live here no better than the
    ants. come with me and i will show you how to live. so they went to the town and
    the country mouse tasted rich food, but a sudden noise sent them scampering in fear.
    goodbye, said the country mouse, i would rather gnaw a crust in peace than feast in
    fear.
""".trimIndent().replace("\n", " ").replace(Regex(" +"), " ")

/** Adam optimizer with one moment-pair of state per parameter tensor. */
class Adam(private val params: List<Tensor>, var lr: Double = 3e-3) {
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

/**
 * Learning-rate schedule: linear **warmup** for `warmup` steps (0 → peak), then
 * optional **cosine decay** from peak down to `minLr` over the remaining steps.
 * Warmup avoids a big early step wrecking freshly-initialized weights; cosine
 * decay anneals the rate so late training settles into a good minimum.
 * `stepIdx` is 0-based.
 */
fun scheduledLr(stepIdx: Int, totalSteps: Int, peakLr: Double, warmup: Int, cosine: Boolean, minLr: Double): Double {
    if (warmup > 0 && stepIdx < warmup) return peakLr * (stepIdx + 1).toDouble() / warmup
    if (!cosine) return peakLr
    val denom = maxOf(1, totalSteps - warmup)
    val t = ((stepIdx - warmup).toDouble() / denom).coerceIn(0.0, 1.0)
    return minLr + 0.5 * (peakLr - minLr) * (1.0 + Math.cos(Math.PI * t))
}

/**
 * Clip the global gradient (L2) norm across all parameters to `maxNorm`, in
 * place. If the combined norm exceeds the cap, every gradient is scaled down by
 * the same factor — this preserves direction while bounding the step size, which
 * tames the occasional exploding gradient on bigger/deeper models. Returns the
 * pre-clip norm. `maxNorm <= 0` disables clipping.
 */
fun clipGradNorm(params: List<Tensor>, maxNorm: Double): Double {
    var sq = 0.0
    for (p in params) for (g in p.grad) sq += g * g
    val norm = sqrt(sq)
    if (maxNorm > 0.0 && norm > maxNorm) {
        val scale = maxNorm / (norm + 1e-6)
        for (p in params) for (i in p.grad.indices) p.grad[i] *= scale
    }
    return norm
}

/**
 * Autoregressive generation (sliding window): feed the model its own output one
 * token at a time. Recomputes the full context each step; use `generateKV`
 * (Inference.kt) for the faster KV-cache path within one context window.
 */
fun generate(model: GPT, tok: Tokenizer, prompt: String, maxNew: Int, sampler: Sampler, rng: Random): String {
    var ids = tok.encode(prompt.ifEmpty { " " })
    for (step in 0 until maxNew) {
        val context = if (ids.size <= model.cfg.blockSize) ids else ids.copyOfRange(ids.size - model.cfg.blockSize, ids.size)
        val logits = model.forward(context)
        val last = DoubleArray(model.cfg.vocabSize) { logits.at(context.size - 1, it) }
        ids = ids + sampleFrom(last, sampler, rng)
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
fun computeMaxGradError(useRope: Boolean = false, tie: Boolean = false): Double {
    val text = "hello world, transformers!"
    val tok = CharTokenizer(text)
    // dropout stays 0 here so the forward pass is deterministic for finite diffs.
    val cfg = Config(vocabSize = tok.vocabSize, blockSize = 8, nEmbed = 16, nHead = 2, nLayer = 2, useRope = useRope, tieWeights = tie)
    val model = GPT(cfg, seed = 7)
    val idx = tok.encode("hello wo")
    val tgt = tok.encode("ello wor")

    model.parameters().forEach { it.grad.fill(0.0) }
    model.loss(idx, tgt).backward()

    val eps = 1e-5
    val rng = Random(0)
    var maxRel = 0.0
    // Spot-check a handful of random parameters across a few tensors. tokEmb is
    // always included — when tied it also serves as the output projection, so it
    // exercises both gradient paths.
    val toCheck = listOfNotNull(model.head, model.tokEmb, model.blocks[0].attn.wq, model.blocks[1].mlp.w1, model.lnFg)
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
    val cases = listOf(
        Triple("learned pos", false, false),
        Triple("RoPE", true, false),
        Triple("tied weights", false, true),
    )
    for ((label, useRope, tie) in cases) {
        val maxRel = computeMaxGradError(useRope, tie)
        val verdict = if (maxRel < 1e-4) "PASS" else "FAIL"
        println("Gradient check [%-12s]: max relative error = %.2e  $verdict".format(label, maxRel))
    }
    println()
}

/** Build a tokenizer of the requested kind over the corpus. */
fun makeTokenizer(kind: String, text: String): Tokenizer = when (kind) {
    "bpe" -> BpeTokenizer.train(text, targetVocab = 300)
    else -> CharTokenizer(text)
}

/**
 * Mean cross-entropy on held-out data, evaluated in eval mode (no dropout).
 * Uses `count` evenly-spaced windows over data[from ..], so it's deterministic
 * and comparable across training steps. Returns NaN if the region is too small.
 */
fun heldOutLoss(model: GPT, data: IntArray, from: Int, block: Int, count: Int): Double {
    val span = data.size - from - block - 1      // last valid window start offset
    if (span < 0) return Double.NaN
    val k = minOf(count, span + 1)
    var sum = 0.0
    for (w in 0 until k) {
        val start = from + if (k == 1) 0 else w * span / (k - 1)
        val idx = data.copyOfRange(start, start + block)
        val tgt = data.copyOfRange(start + 1, start + block + 1)
        sum += Tensor.crossEntropy(model.forward(idx), tgt).data[0]   // train=false -> no dropout
    }
    return sum / k
}

/**
 * Mean per-token cross-entropy over a whole token sequence, in eval mode. Sweeps
 * windows at the given `stride` (default = block, i.e. non-overlapping chunks).
 * `perplexity = exp(meanLoss)` — the standard language-model quality metric: the
 * effective number of equally-likely choices the model is deciding between.
 */
fun corpusLoss(model: GPT, data: IntArray, block: Int, stride: Int): Double {
    require(data.size > block) { "text shorter than the model's context ($block)" }
    val step = maxOf(1, stride)
    var sum = 0.0; var count = 0
    var start = 0
    while (start + block + 1 <= data.size) {
        val idx = data.copyOfRange(start, start + block)
        val tgt = data.copyOfRange(start + 1, start + block + 1)
        sum += Tensor.crossEntropy(model.forward(idx), tgt).data[0]
        count++
        start += step
    }
    return if (count == 0) Double.NaN else sum / count
}

/**
 * Train a model on `text` with `tok`; returns the trained model.
 *
 * The last `valFraction` of the corpus is held out for validation. Training only
 * samples windows from the train region; every `evalEvery` steps we report the
 * held-out loss. Watching val loss stop improving (or rise) while train loss
 * keeps falling is exactly what overfitting looks like.
 *
 * If `saveBest` is set (and there's a validation split), the model is checkpointed
 * to that path every time held-out loss reaches a new low — i.e. early stopping:
 * the file ends up holding the best-generalizing weights, not the last ones.
 */
fun trainModel(
    tok: Tokenizer, text: String,
    steps: Int, batch: Int = 12, lr: Double = 3e-3,
    cfg: Config, seed: Long = 1234L, verbose: Boolean = true,
    valFraction: Double = 0.1, evalEvery: Int = 250, evalWindows: Int = 64,
    warmup: Int = 0, cosine: Boolean = false, minLr: Double = 0.0, clip: Double = 0.0,
    saveBest: String? = null,
): GPT {
    val data = tok.encode(text)
    val model = GPT(cfg, seed)
    val opt = Adam(model.parameters(), lr)
    val rng = Random(seed + 999)   // batch order depends on seed, so runs differ
    require(data.size > cfg.blockSize + 1) { "corpus too short for blockSize=${cfg.blockSize}" }

    // Split off a held-out tail, but only if both halves can hold a window.
    val split = (data.size * (1.0 - valFraction)).toInt()
    val hasVal = valFraction > 0.0 &&
        split > cfg.blockSize + 1 && (data.size - split) > cfg.blockSize + 1
    val trainEnd = if (hasVal) split else data.size

    if (verbose) {
        println(
            "chars=${text.length}  tokens=${data.size}  vocab=${tok.vocabSize}  " +
                "params=${model.parameters().sumOf { it.data.size }}"
        )
        if (hasVal) println("split: train=${trainEnd} tokens, val=${data.size - trainEnd} tokens")
        else println("split: corpus too small to hold out a validation set (train on all)")
    }

    var running = 0.0
    var lastNorm = 0.0
    var bestVal = Double.POSITIVE_INFINITY
    var bestStep = -1
    for (step in 1..steps) {
        var lossVal = 0.0
        opt.zeroGrad()
        for (b in 0 until batch) {
            val start = rng.nextInt(trainEnd - cfg.blockSize - 1)
            val idx = data.copyOfRange(start, start + cfg.blockSize)
            val tgt = data.copyOfRange(start + 1, start + cfg.blockSize + 1)
            val l = model.loss(idx, tgt)
            l.backward()               // gradients accumulate across the batch
            lossVal += l.data[0]
        }
        model.parameters().forEach { p -> for (i in p.grad.indices) p.grad[i] /= batch }
        lastNorm = clipGradNorm(model.parameters(), clip)          // bound the step size
        opt.lr = scheduledLr(step - 1, steps, lr, warmup, cosine, minLr)
        opt.step()

        running += lossVal / batch
        if (evalEvery > 0 && step % evalEvery == 0) {
            val trainLoss = running / evalEvery
            running = 0.0
            val valLoss = if (hasVal) heldOutLoss(model, data, trainEnd, cfg.blockSize, evalWindows) else Double.NaN

            // Early stopping: keep the checkpoint at its lowest held-out loss.
            var improved = false
            if (hasVal && saveBest != null && valLoss < bestVal) {
                bestVal = valLoss; bestStep = step; improved = true
                Checkpoint.save(saveBest, cfg, model, tok)
            }
            if (verbose) {
                val head = "step %5d  lr %.2e  gnorm %5.2f  train %.4f".format(step, opt.lr, lastNorm, trainLoss)
                if (hasVal) println("$head  val %.4f  (gap %+.4f)%s".format(valLoss, valLoss - trainLoss, if (improved) "  * saved" else ""))
                else println(head)
            }
        }
    }
    if (verbose && bestStep > 0) println("best val %.4f at step %d -> %s".format(bestVal, bestStep, saveBest))
    return model
}

fun train(kind: String, useRope: Boolean) {
    val text = corpus()
    val tok = makeTokenizer(kind, text)
    val blockSize = if (kind == "bpe") 16 else 24   // BPE tokens cover more text per step
    val cfg = Config(vocabSize = tok.vocabSize, blockSize = blockSize, nEmbed = 64, nHead = 4, nLayer = 2, useRope = useRope)

    println("tokenizer=$kind  positions=${if (useRope) "rope" else "learned"}")
    val model = trainModel(tok, text, steps = 1500, cfg = cfg)

    val rng = Random(7)
    println("\n--- samples (temperature 0.8) ---")
    for (prompt in listOf("to be", "the ", "knowledge")) {
        println("[$prompt] -> ${generate(model, tok, prompt, maxNew = 60, sampler = Sampler(0.8), rng = rng)}")
    }
}

/** Train briefly, then render attention heatmaps for a sample string. */
fun attnMode(kind: String, useRope: Boolean) {
    val text = corpus()
    val tok = makeTokenizer(kind, text)
    val blockSize = if (kind == "bpe") 16 else 24
    val cfg = Config(vocabSize = tok.vocabSize, blockSize = blockSize, nEmbed = 48, nHead = 3, nLayer = 2, useRope = useRope)

    println("tokenizer=$kind  positions=${if (useRope) "rope" else "learned"} — training before visualizing...")
    val model = trainModel(tok, text, steps = 800, cfg = cfg)

    val probe = "the quick brown".take(blockSize)
    visualizeAttention(model, tok, probe, htmlPath = "attention.html")
}

/** Mean and sample standard deviation of a list. */
fun meanStd(xs: List<Double>): Pair<Double, Double> {
    val m = xs.average()
    val v = if (xs.size > 1) xs.sumOf { (it - m) * (it - m) } / (xs.size - 1) else 0.0
    return m to sqrt(v)
}

/**
 * Train `runs` models of `cfg` (each with a different seed) with early stopping,
 * and return every run's best held-out loss. Averaging across seeds is what makes
 * a comparison trustworthy — a single run is noisy.
 */
fun bestValRuns(tok: Tokenizer, text: String, cfg: Config, steps: Int, runs: Int, valFraction: Double = 0.1): List<Double> {
    val data = tok.encode(text)
    val split = (data.size * (1.0 - valFraction)).toInt()
    val out = ArrayList<Double>(runs)
    for (r in 0 until runs) {
        val tmp = File.createTempFile("kortex-run", ".bin").apply { deleteOnExit() }
        trainModel(tok, text, steps = steps, cfg = cfg, seed = 1000L + r, verbose = false,
            valFraction = valFraction, evalEvery = maxOf(1, steps / 10), saveBest = tmp.path)
        val (best, _) = Checkpoint.load(tmp.path)
        out.add(heldOutLoss(best, data, split, cfg.blockSize, 128))
    }
    return out
}

/** Train learned-absolute vs. RoPE (averaged over --runs) and compare held-out loss. */
fun posCompare(cli: Cli) {
    val kind = cli.pos(1, "char")
    val text = if (cli.strOrNull("data") != null) File(cli.strOrNull("data")!!).readText() else sampleCorpus()
    val tok = makeTokenizer(kind, text)
    val block = cli.int("block", if (kind == "bpe") 16 else 24)
    val steps = cli.int("steps", 1000)
    val runs = cli.int("runs", 3)

    println("learned vs. RoPE  (tok=$kind, $steps steps, $runs runs, early-stopped)\n")
    println("  %-18s  %-18s  %-10s  %s".format("positions", "best val (mean±sd)", "perplexity", "params"))
    for (useRope in listOf(false, true)) {
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = block, nEmbed = cli.int("embed", 64),
            nHead = cli.int("heads", 4), nLayer = cli.int("layers", 2), useRope = useRope)
        val (m, sd) = meanStd(bestValRuns(tok, text, cfg, steps, runs))
        val name = if (useRope) "RoPE (rotary)" else "learned absolute"
        println("  %-18s  %.4f ± %.4f     %-10.2f  %d".format(name, m, sd, Math.exp(m), GPT(cfg).parameters().sumOf { it.data.size }))
    }
    println("\nRoPE uses fewer params (no position table) and encodes *relative* position.")
}

/* ----------------------------- CLI ----------------------------- */

/** Minimal argument parser: `<cmd> [positional...] [--key value] [--flag]`. */
class Cli(argv: Array<String>) {
    val cmd = argv.getOrNull(0) ?: "help"
    val positionals = ArrayList<String>()
    private val opts = HashMap<String, String>()
    private val bools = HashSet<String>()

    init {
        var i = 1
        while (i < argv.size) {
            val a = argv[i]
            if (a.startsWith("--")) {
                val key = a.substring(2)
                val next = argv.getOrNull(i + 1)
                if (next != null && !next.startsWith("--")) { opts[key] = next; i += 2 }
                else { bools.add(key); i += 1 }
            } else { positionals.add(a); i += 1 }
        }
    }

    fun str(k: String, d: String) = opts[k] ?: d
    fun strOrNull(k: String) = opts[k]
    fun int(k: String, d: Int) = opts[k]?.toInt() ?: d
    fun long(k: String, d: Long) = opts[k]?.toLong() ?: d
    fun dbl(k: String, d: Double) = opts[k]?.toDouble() ?: d
    fun flag(k: String) = k in bools
    /** n-th positional after the command (1-based), or default. */
    fun pos(n: Int, d: String) = positionals.getOrNull(n - 1) ?: d
}

/** train: fit a model on the built-in corpus or a text file, then save it. */
fun cmdTrain(cli: Cli) {
    val dataPath = cli.strOrNull("data")
    val text = when {
        dataPath != null -> File(dataPath).readText()
        cli.flag("sample") -> sampleCorpus()
        else -> corpus()
    }
    val kind = cli.str("tok", cli.pos(1, "char"))
    val useRope = cli.flag("rope")
    val tok = makeTokenizer(kind, text)
    val cfg = Config(
        vocabSize = tok.vocabSize,
        blockSize = cli.int("block", if (kind == "bpe") 16 else 24),
        nEmbed = cli.int("embed", 64),
        nHead = cli.int("heads", 4),
        nLayer = cli.int("layers", 2),
        useRope = useRope,
        tieWeights = cli.flag("tie"),
        dropout = cli.dbl("dropout", 0.0),
    )
    val out = cli.str("out", "model.bin")
    println("training: tok=$kind rope=$useRope tie=${cfg.tieWeights} dropout=${cfg.dropout} embed=${cfg.nEmbed} heads=${cfg.nHead} layers=${cfg.nLayer} block=${cfg.blockSize} params=${GPT(cfg).parameters().sumOf { it.data.size }}")
    val peakLr = cli.dbl("lr", 3e-3)
    val earlyStop = cli.flag("early-stop")
    val model = trainModel(
        tok, text,
        steps = cli.int("steps", 1500), batch = cli.int("batch", 12),
        lr = peakLr, cfg = cfg,
        valFraction = cli.dbl("val", 0.1), evalEvery = cli.int("eval-every", 250),
        warmup = cli.int("warmup", 0), cosine = cli.flag("cosine"),
        minLr = cli.dbl("min-lr", peakLr * 0.1), clip = cli.dbl("clip", 0.0),
        saveBest = if (earlyStop) out else null,
    )
    // With early stopping the best-val checkpoint is already on disk; otherwise
    // save the final model. (If there was no val split, ensure a file exists.)
    if (!earlyStop) Checkpoint.save(out, cfg, model, tok)
    else if (!File(out).exists()) { Checkpoint.save(out, cfg, model, tok); println("(no val split — saved final model)") }
    println("\nsaved model -> $out  (use: generate --model $out --prompt \"...\"  or  chat --model $out)")
    runCatching {
        // Sample from the model on disk (the best one when early-stopping).
        val (best, bestTok) = Checkpoint.load(out)
        val seed = if (dataPath == null && !cli.flag("sample")) "to be" else "the "
        println("--- sample ---")
        println(generate(best, bestTok, seed, maxNew = 80, sampler = Sampler(0.8), rng = Random(7)))
    }.onFailure { println("(sample skipped: ${it.message})") }
}

/** eval: report loss and perplexity of a saved model on a text file. */
fun cmdEval(cli: Cli) {
    val path = cli.str("model", "model.bin")
    if (!File(path).exists()) { System.err.println("No model at '$path'. Train one first."); return }
    val dataPath = cli.strOrNull("data")
    if (dataPath == null) { System.err.println("eval needs --data <file> to score."); return }
    val (model, tok) = Checkpoint.load(path)
    val text = File(dataPath).readText()
    val data = tok.encode(text)
    val block = model.cfg.blockSize
    if (data.size <= block) { System.err.println("text has ${data.size} tokens, need > $block (the context)."); return }
    val stride = cli.int("stride", block)
    val loss = corpusLoss(model, data, block, stride)
    println("eval: tokens=${data.size} block=$block stride=$stride")
    println("cross-entropy (nats/token): %.4f".format(loss))
    println("perplexity               : %.2f".format(Math.exp(loss)))
}

/** Serialize a char-tokenizer model to compact JSON for the browser UI. */
fun exportJson(model: GPT, tok: CharTokenizer): String {
    val cfg = model.cfg
    fun arr(t: Tensor) = t.data.joinToString(",", "[", "]") { it.toString() }
    fun jstr(s: String) = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"")
        .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + "\""
    val sb = StringBuilder("{")
    sb.append("\"vocab\":").append(jstr(tok.chars.joinToString(""))).append(",")
    sb.append("\"blockSize\":${cfg.blockSize},\"nEmbed\":${cfg.nEmbed},\"nHead\":${cfg.nHead},")
    sb.append("\"nLayer\":${cfg.nLayer},\"useRope\":${cfg.useRope},\"tieWeights\":${cfg.tieWeights},")
    sb.append("\"tokEmb\":").append(arr(model.tokEmb)).append(",")
    sb.append("\"posEmb\":").append(model.posEmb?.let { arr(it) } ?: "null").append(",")
    sb.append("\"lnFg\":").append(arr(model.lnFg)).append(",\"lnFb\":").append(arr(model.lnFb)).append(",")
    sb.append("\"head\":").append(model.head?.let { arr(it) } ?: "null").append(",")
    sb.append("\"blocks\":[")
    model.blocks.forEachIndexed { i, b ->
        if (i > 0) sb.append(",")
        sb.append("{\"ln1g\":").append(arr(b.ln1g)).append(",\"ln1b\":").append(arr(b.ln1b))
        sb.append(",\"ln2g\":").append(arr(b.ln2g)).append(",\"ln2b\":").append(arr(b.ln2b))
        sb.append(",\"wq\":").append(arr(b.attn.wq)).append(",\"wk\":").append(arr(b.attn.wk))
        sb.append(",\"wv\":").append(arr(b.attn.wv)).append(",\"wo\":").append(arr(b.attn.wo))
        sb.append(",\"w1\":").append(arr(b.mlp.w1)).append(",\"b1\":").append(arr(b.mlp.b1))
        sb.append(",\"w2\":").append(arr(b.mlp.w2)).append(",\"b2\":").append(arr(b.mlp.b2)).append("}")
    }
    sb.append("]}")
    return sb.toString()
}

/** export: write a saved char model to JSON so the browser UI can run it. */
fun cmdExport(cli: Cli) {
    val path = cli.str("model", "model.bin")
    if (!File(path).exists()) { System.err.println("No model at '$path'."); return }
    val (model, tok) = Checkpoint.load(path)
    if (tok !is CharTokenizer) { System.err.println("export supports the char tokenizer only (train without --tok bpe)."); return }
    val out = cli.str("out", "model.json")
    File(out).writeText(exportJson(model, tok))
    println("exported ${File(out).length()} bytes -> $out  (embed in the web chat UI)")
}

/** tiecompare: train tied vs. untied under identical settings; compare held-out quality. */
fun tieCompare(cli: Cli) {
    val text = when {
        cli.strOrNull("data") != null -> File(cli.strOrNull("data")!!).readText()
        else -> sampleCorpus()   // large enough to hold out a validation set
    }
    val kind = cli.str("tok", "char")
    val tok = makeTokenizer(kind, text)
    val data = tok.encode(text)
    val block = cli.int("block", if (kind == "bpe") 16 else 24)
    val steps = cli.int("steps", 1000)
    val valFraction = 0.1
    val split = (data.size * (1.0 - valFraction)).toInt()
    require(data.size - split > block + 1) { "corpus too small for a val split; use --data" }

    // Compare each model at its BEST held-out point (early stopping), averaged
    // over --runs seeds so the verdict isn't a single-run fluke.
    val runs = cli.int("runs", 3)
    println("learned vs. tied  (tok=$kind, $steps steps, $runs runs, held-out ${data.size - split} tokens)\n")
    println("  %-18s  %-18s  %-10s  %s".format("output head", "best val (mean±sd)", "perplexity", "params"))
    val results = LinkedHashMap<String, Double>()
    for (tie in listOf(false, true)) {
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = block, nEmbed = cli.int("embed", 64),
            nHead = cli.int("heads", 4), nLayer = cli.int("layers", 2), tieWeights = tie)
        val (m, sd) = meanStd(bestValRuns(tok, text, cfg, steps, runs, valFraction))
        val name = if (tie) "tied (shared)" else "learned (separate)"
        results[name] = m
        println("  %-18s  %.4f ± %.4f     %-10.2f  %d".format(name, m, sd, Math.exp(m), GPT(cfg).parameters().sumOf { it.data.size }))
    }
    val winner = results.minByOrNull { it.value }!!
    val other = results.maxByOrNull { it.value }!!
    println("\n-> ${winner.key} generalizes better here (mean val %.4f vs %.4f).".format(winner.value, other.value))
    println("   Tying always uses vocab×embed fewer params; whether it also wins on")
    println("   quality varies with corpus and run — try --data, more --steps, more --runs.")
}

/** Build a Sampler from --temp / --top-k / --top-p flags. */
fun samplerFrom(cli: Cli) = Sampler(
    temperature = cli.dbl("temp", 0.8),
    topK = cli.int("top-k", 0),
    topP = cli.dbl("top-p", 1.0),
)

/** generate: load a saved model and continue a prompt once. */
fun cmdGenerate(cli: Cli) {
    val path = cli.str("model", "model.bin")
    if (!File(path).exists()) { System.err.println("No model at '$path'. Train one first: train --out $path"); return }
    val (model, tok) = Checkpoint.load(path)
    val prompt = cli.strOrNull("prompt") ?: cli.positionals.joinToString(" ")
    val sampler = samplerFrom(cli)
    val tokens = cli.int("tokens", 200)
    val rng = Random(cli.long("seed", 42))
    runCatching {
        // --kv uses the KV-cache path (bounded by the context window).
        if (cli.flag("kv")) generateKV(model, tok, prompt, tokens, sampler, rng)
        else generate(model, tok, prompt, tokens, sampler, rng)
    }.onSuccess { println(it) }
        .onFailure { System.err.println("Cannot encode prompt (a character isn't in the model's vocab): ${it.message}") }
}

/** chat: interactive REPL — type a prompt, the model continues it. */
fun cmdChat(cli: Cli) {
    val path = cli.str("model", "model.bin")
    if (!File(path).exists()) { System.err.println("No model at '$path'. Train one first: train --out $path"); return }
    val (model, tok) = Checkpoint.load(path)
    var tokens = cli.int("tokens", 120)
    var sampler = samplerFrom(cli)
    val useKv = cli.flag("kv")
    val rng = Random(cli.long("seed", 42))
    println("Kortex chat — type a prompt; the model continues it. (A tiny char/BPE LM, not an assistant.)")
    println("Commands:  :temp <x>   :top-k <n>   :top-p <x>   :tokens <n>   :quit")
    while (true) {
        print("\n> "); System.out.flush()
        val line = readLine() ?: break
        val s = line.trim()
        when {
            s == ":quit" || s == ":q" -> break
            s.isEmpty() -> {}
            s.startsWith(":temp") -> { sampler = sampler.copy(temperature = s.removePrefix(":temp").trim().toDoubleOrNull() ?: sampler.temperature); println("$sampler") }
            s.startsWith(":top-k") -> { sampler = sampler.copy(topK = s.removePrefix(":top-k").trim().toIntOrNull() ?: sampler.topK); println("$sampler") }
            s.startsWith(":top-p") -> { sampler = sampler.copy(topP = s.removePrefix(":top-p").trim().toDoubleOrNull() ?: sampler.topP); println("$sampler") }
            s.startsWith(":tokens") -> { tokens = s.removePrefix(":tokens").trim().toIntOrNull() ?: tokens; println("tokens=$tokens") }
            else -> runCatching { if (useKv) generateKV(model, tok, s, tokens, sampler, rng) else generate(model, tok, s, tokens, sampler, rng) }
                .onSuccess { println(it) }
                .onFailure { println("(can't encode that — a character isn't in the model's vocab)") }
        }
    }
    println("bye")
}

/** bench: measure KV-cache vs. full-recompute decoding on a random-init model. */
fun cmdBench(cli: Cli) {
    val cfg = Config(
        vocabSize = cli.int("vocab", 96),
        blockSize = cli.int("block", 128),
        nEmbed = cli.int("embed", 128),
        nHead = cli.int("heads", 4),
        nLayer = cli.int("layers", 4),
        useRope = cli.flag("rope"),
    )
    val model = GPT(cfg, seed = 1)
    val n = cfg.blockSize
    println("bench: block=$n embed=${cfg.nEmbed} heads=${cfg.nHead} layers=${cfg.nLayer} rope=${cfg.useRope}")
    println("decoding $n tokens (greedy)...")

    // Full recompute: re-run the whole context every step.
    var t0 = System.nanoTime()
    val full = ArrayList<Int>(); full.add(0)
    while (full.size < n) {
        val logits = model.forward(full.toIntArray())
        val last = DoubleArray(cfg.vocabSize) { logits.at(full.size - 1, it) }
        full.add(argmax(last))
    }
    val fullMs = (System.nanoTime() - t0) / 1e6

    // KV-cache: each step is O(t) dot-products over cached keys/values.
    t0 = System.nanoTime()
    val gen = KVGenerator(model)
    val kv = ArrayList<Int>(); kv.add(0)
    var logits = gen.step(0)
    while (gen.length < n) { val nx = argmax(logits); kv.add(nx); if (gen.length >= n) break; logits = gen.step(nx) }
    val kvMs = (System.nanoTime() - t0) / 1e6

    val match = full == kv
    println("full recompute : %8.1f ms  (%.0f tok/s)".format(fullMs, n / (fullMs / 1000)))
    println("KV-cache       : %8.1f ms  (%.0f tok/s)".format(kvMs, n / (kvMs / 1000)))
    println("speedup        : %.1fx".format(fullMs / kvMs))
    println("outputs identical: $match")
}

/** schedule: draw the learning-rate curve (warmup + cosine) as a sparkline. */
fun cmdSchedule(cli: Cli) {
    val steps = cli.int("steps", 1500)
    val warmup = cli.int("warmup", maxOf(1, steps / 10))
    val cosine = !cli.flag("no-cosine")          // curve viz defaults to cosine on
    val peak = cli.dbl("lr", 3e-3)
    val minLr = cli.dbl("min-lr", peak * 0.1)

    val cols = 64
    val ramp = " ▁▂▃▄▅▆▇█"
    val vals = DoubleArray(cols) { c ->
        val s = if (cols == 1) 0 else c * (steps - 1) / (cols - 1)
        scheduledLr(s, steps, peak, warmup, cosine, minLr)
    }
    val hi = vals.max(); val lo = vals.min()
    val bars = buildString {
        for (v in vals) {
            val norm = if (hi > lo) (v - lo) / (hi - lo) else 0.0
            append(ramp[(norm * (ramp.length - 1)).toInt().coerceIn(0, ramp.length - 1)])
        }
    }
    println("LR schedule: steps=$steps warmup=$warmup cosine=$cosine peak=%.2e min=%.2e".format(peak, minLr))
    println(bars)
    println("^peak reached at step $warmup, then ${if (cosine) "cosine-decays to" else "holds at"} %.2e".format(vals.last()))
}

fun printHelp() {
    println(
        """
        Kortex — a minimalistic LLM in Kotlin.

        Usage: <command> [options]

        Commands:
          train        Train a model and save it.
            --data <file>     text corpus (default: built-in toy corpus)
            --sample          use the larger built-in corpus (Aesop, ~2 KB)
            --val F           held-out validation fraction (default: 0.1)
            --eval-every N    report train/val loss every N steps (default: 250)
            --tok char|bpe    tokenizer (default: char)
            --rope            rotary positions instead of a learned table
            --tie             tie the output projection to the token embedding
            --dropout F       dropout probability, training only (default: 0)
            --embed N (64)    --heads N (4)   --layers N (2)   --block N
            --steps N (1500)  --batch N (12)  --lr F (0.003)
            --warmup N        linear LR warmup steps (default: 0)
            --cosine          cosine-decay the LR after warmup   --min-lr F
            --clip F          clip global gradient norm to F (default: off)
            --early-stop      save the checkpoint at its lowest validation loss
            --out <file>      checkpoint path (default: model.bin)

          eval         Report loss and perplexity of a saved model on a text file.
            --model <file> (model.bin)  --data <file> (required)  --stride N

          tiecompare   Train tied vs. untied and compare held-out perplexity.
            --data <file> | (built-in sample)  --steps N  --runs N (3)  --embed/...

          export       Write a saved char model to JSON for the browser chat UI.
            --model <file> (model.bin)  --out <file> (model.json)

          generate     Continue a prompt once, using a saved model.
            --model <file> (model.bin)  --prompt "text"  --tokens N (200)
            --temp F (0.8)   --top-k N (off)   --top-p F (off)   --seed N (42)
            --kv            use the KV-cache path (bounded by the context window)

          chat         Interactive REPL over a saved model (reads stdin).
            --model <file> (model.bin)  --tokens N (120)  --kv
            --temp F   --top-k N   --top-p F   (also as :commands inside chat)

          bench        KV-cache vs. full-recompute decoding speed (random model).
            --block N (128)  --embed N (128)  --heads N (4)  --layers N (4)  --rope

          schedule     Draw the LR warmup+cosine curve as a sparkline.
            --steps N (1500)  --warmup N  --lr F  --min-lr F  --no-cosine

          gradcheck    Verify backprop (learned + RoPE).
          attn [char|bpe] [--rope]      Train small, visualize attention -> attention.html
          poscompare [char|bpe]         Compare learned vs. RoPE positions
          help         Show this message.
        """.trimIndent()
    )
}

fun main(args: Array<String>) {
    val cli = Cli(args)
    when (cli.cmd) {
        "train" -> cmdTrain(cli)
        "generate" -> cmdGenerate(cli)
        "chat" -> cmdChat(cli)
        "eval" -> cmdEval(cli)
        "tiecompare" -> tieCompare(cli)
        "export" -> cmdExport(cli)
        "bench" -> cmdBench(cli)
        "schedule" -> cmdSchedule(cli)
        "gradcheck" -> gradCheck()
        "attn" -> attnMode(cli.pos(1, "char"), cli.flag("rope"))
        "poscompare" -> posCompare(cli)
        "help", "--help", "-h" -> printHelp()
        else -> { gradCheck(); train("char", false) }   // no-arg default demo
    }
}
