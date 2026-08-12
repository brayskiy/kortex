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
 * Train a model on `text` with `tok`; returns the trained model.
 *
 * The last `valFraction` of the corpus is held out for validation. Training only
 * samples windows from the train region; every `evalEvery` steps we report the
 * held-out loss. Watching val loss stop improving (or rise) while train loss
 * keeps falling is exactly what overfitting looks like.
 */
fun trainModel(
    tok: Tokenizer, text: String,
    steps: Int, batch: Int = 12, lr: Double = 3e-3,
    cfg: Config, seed: Long = 1234L, verbose: Boolean = true,
    valFraction: Double = 0.1, evalEvery: Int = 250, evalWindows: Int = 64,
): GPT {
    val data = tok.encode(text)
    val model = GPT(cfg, seed)
    val opt = Adam(model.parameters(), lr)
    val rng = Random(42)
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
        opt.step()

        running += lossVal / batch
        if (verbose && evalEvery > 0 && step % evalEvery == 0) {
            val trainLoss = running / evalEvery
            running = 0.0
            if (hasVal) {
                val valLoss = heldOutLoss(model, data, trainEnd, cfg.blockSize, evalWindows)
                val gap = valLoss - trainLoss
                println("step %5d  train %.4f  val %.4f  (gap %+.4f)".format(step, trainLoss, valLoss, gap))
            } else {
                println("step %5d  train %.4f".format(step, trainLoss))
            }
        }
    }
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

/** Train learned-absolute vs. RoPE with identical settings and compare. */
fun posCompare(kind: String) {
    val text = corpus()
    val tok = makeTokenizer(kind, text)
    val blockSize = if (kind == "bpe") 16 else 24
    val data = tok.encode(text)
    val idx = data.copyOfRange(0, blockSize)
    val tgt = data.copyOfRange(1, blockSize + 1)

    println("Positional-encoding comparison (tokenizer=$kind, 1000 steps each)\n")
    println("  %-18s  %-10s  %s".format("encoding", "eval loss", "params"))
    for (useRope in listOf(false, true)) {
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = blockSize, nEmbed = 64, nHead = 4, nLayer = 2, useRope = useRope)
        val model = trainModel(tok, text, steps = 1000, cfg = cfg, verbose = false)
        val loss = model.loss(idx, tgt).data[0]
        val name = if (useRope) "RoPE (rotary)" else "learned absolute"
        println("  %-18s  %-10.4f  %d".format(name, loss, model.parameters().sumOf { it.data.size }))
    }
    println("\nRoPE uses fewer params (no ${blockSize}x64 position table) and generalizes")
    println("to longer contexts because it encodes *relative* position.")
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
    val model = trainModel(
        tok, text,
        steps = cli.int("steps", 1500), batch = cli.int("batch", 12),
        lr = cli.dbl("lr", 3e-3), cfg = cfg,
        valFraction = cli.dbl("val", 0.1), evalEvery = cli.int("eval-every", 250),
    )
    Checkpoint.save(out, cfg, model, tok)
    println("\nsaved model -> $out  (use: generate --model $out --prompt \"...\"  or  chat --model $out)")
    runCatching {
        val seed = if (dataPath == null && !cli.flag("sample")) "to be" else "the "
        println("--- sample ---")
        println(generate(model, tok, seed, maxNew = 80, sampler = Sampler(0.8), rng = Random(7)))
    }.onFailure { println("(sample skipped: ${it.message})") }
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
            --out <file>      checkpoint path (default: model.bin)

          generate     Continue a prompt once, using a saved model.
            --model <file> (model.bin)  --prompt "text"  --tokens N (200)
            --temp F (0.8)   --top-k N (off)   --top-p F (off)   --seed N (42)
            --kv            use the KV-cache path (bounded by the context window)

          chat         Interactive REPL over a saved model (reads stdin).
            --model <file> (model.bin)  --tokens N (120)  --kv
            --temp F   --top-k N   --top-p F   (also as :commands inside chat)

          bench        KV-cache vs. full-recompute decoding speed (random model).
            --block N (128)  --embed N (128)  --heads N (4)  --layers N (4)  --rope

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
        "bench" -> cmdBench(cli)
        "gradcheck" -> gradCheck()
        "attn" -> attnMode(cli.pos(1, "char"), cli.flag("rope"))
        "poscompare" -> posCompare(cli.pos(1, "char"))
        "help", "--help", "-h" -> printHelp()
        else -> { gradCheck(); train("char", false) }   // no-arg default demo
    }
}
