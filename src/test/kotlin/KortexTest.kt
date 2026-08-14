/*
 * KortexTest.kt — automated checks run by `./gradlew test`.
 *
 * These assert the properties the whole thing depends on:
 *   - backprop matches finite-difference gradients (the math is right)
 *   - tokenizers are lossless (decode ∘ encode == identity)
 *   - BPE actually compresses the sequence
 *   - attention is causal and each row is a probability distribution
 *   - training reduces the loss
 */

import java.io.File
import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class KortexTest {

    /** top-k=1 and top-p→0 must both collapse to greedy (the argmax token). */
    @Test
    fun topKAndTopPRestrictToArgmax() {
        val logits = doubleArrayOf(0.1, 3.0, 0.2, 2.9, -1.0)   // argmax = index 1
        val rng = Random(123)
        repeat(200) {
            assertEquals(1, sampleFrom(logits, Sampler(temperature = 1.0, topK = 1), rng))
            assertEquals(1, sampleFrom(logits, Sampler(temperature = 1.0, topP = 1e-6), rng))
        }
    }

    /** top-k must never return a token outside the k highest-logit set. */
    @Test
    fun topKNeverPicksOutsideTheTopK() {
        val logits = doubleArrayOf(5.0, 4.0, -2.0, -3.0, 0.0, 1.0)
        val allowed = setOf(0, 1, 5)   // three largest
        val rng = Random(7)
        repeat(500) {
            val pick = sampleFrom(logits, Sampler(temperature = 1.0, topK = 3), rng)
            assertTrue(pick in allowed, "top-k picked $pick outside $allowed")
        }
    }

    /** Backprop stays correct with weights tied (tokEmb feeds two paths). */
    @Test
    fun backpropMatchesNumericalGradients_tied() {
        assertTrue(computeMaxGradError(tie = true) < 1e-4)
    }

    /** Tying removes exactly the separate output head (vocab x nEmbed params). */
    @Test
    fun weightTyingDropsTheHead() {
        val base = Config(vocabSize = 30, blockSize = 8, nEmbed = 16, nHead = 2, nLayer = 1)
        val tied = base.copy(tieWeights = true)
        val pBase = GPT(base, seed = 1).parameters().sumOf { it.data.size }
        val pTied = GPT(tied, seed = 1).parameters().sumOf { it.data.size }
        assertEquals(base.vocabSize * base.nEmbed, pBase - pTied)
        assertEquals(null, GPT(tied, seed = 1).head)
    }

    /** A tied model must round-trip through a checkpoint too. */
    @Test
    fun tiedCheckpointRoundTrips() {
        val tok = CharTokenizer(corpus())
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = 8, nEmbed = 16, nHead = 2, nLayer = 1, tieWeights = true)
        val model = trainModel(tok, corpus(), steps = 20, cfg = cfg, verbose = false)
        val ids = tok.encode(corpus()).copyOfRange(0, cfg.blockSize)
        val before = model.forward(ids)
        val file = File.createTempFile("kortex-tied", ".bin").apply { deleteOnExit() }
        Checkpoint.save(file.path, cfg, model, tok)
        val (loaded, _) = Checkpoint.load(file.path)
        val after = loaded.forward(ids)
        for (i in before.data.indices) assertTrue(Math.abs(before.data[i] - after.data[i]) < 1e-12)
    }

    /** Dropout's backward must match finite differences (fixed mask via reseed). */
    @Test
    fun dropoutBackpropIsCorrect() {
        val n = 8
        val x = Tensor(1, n)
        val init = Random(1); for (i in 0 until n) x.data[i] = init.nextGaussian()
        val ones = Tensor(n, 1).also { it.data.fill(1.0) }
        // Reseeding Random(42) each call makes the mask identical -> deterministic.
        fun loss(): Tensor = Tensor.dropout(x, 0.5, training = true, rng = Random(42)) matmul ones

        x.grad.fill(0.0); loss().backward()
        val eps = 1e-5; var maxRel = 0.0
        for (i in 0 until n) {
            val o = x.data[i]
            x.data[i] = o + eps; val lp = loss().data[0]
            x.data[i] = o - eps; val lm = loss().data[0]
            x.data[i] = o
            val num = (lp - lm) / (2 * eps)
            maxRel = maxOf(maxRel, Math.abs(x.grad[i] - num) / (Math.abs(x.grad[i]) + Math.abs(num) + 1e-9))
        }
        assertTrue(maxRel < 1e-4, "dropout gradient error too high: $maxRel")
    }

    /** corpusLoss over a single window equals that window's cross-entropy. */
    @Test
    fun corpusLossMatchesSingleWindow() {
        val tok = CharTokenizer(sampleCorpus())
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = 16, nEmbed = 16, nHead = 2, nLayer = 1)
        val model = GPT(cfg, seed = 4)
        val data = tok.encode(sampleCorpus()).copyOfRange(0, cfg.blockSize + 1)   // exactly one window
        val direct = Tensor.crossEntropy(model.forward(data.copyOfRange(0, cfg.blockSize)), data.copyOfRange(1, cfg.blockSize + 1)).data[0]
        val loss = corpusLoss(model, data, cfg.blockSize, stride = cfg.blockSize)
        assertEquals(direct, loss, 1e-12)
        assertEquals(Math.exp(direct), Math.exp(loss), 1e-9)   // perplexity = exp(loss)
    }

    /** A trained model must beat the uniform baseline (perplexity < vocab size). */
    @Test
    fun trainedPerplexityBeatsUniform() {
        val tok = CharTokenizer(sampleCorpus())
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = 16, nEmbed = 32, nHead = 2, nLayer = 1)
        val model = trainModel(tok, sampleCorpus(), steps = 200, cfg = cfg, verbose = false, evalEvery = 0)
        val data = tok.encode(sampleCorpus())
        val ppl = Math.exp(corpusLoss(model, data, cfg.blockSize, stride = cfg.blockSize))
        assertTrue(ppl < tok.vocabSize, "perplexity $ppl should beat uniform ${tok.vocabSize}")
    }

    /** Early stopping saves the best-val model; reloaded, it's no worse than final. */
    @Test
    fun earlyStoppingSavesBestValModel() {
        val tok = CharTokenizer(sampleCorpus())
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = 16, nEmbed = 48, nHead = 3, nLayer = 2)
        val data = tok.encode(sampleCorpus())
        val split = (data.size * 0.9).toInt()
        val file = File.createTempFile("kortex-best", ".bin").apply { deleteOnExit() }
        // High LR + no regularization on a small train set => it will overfit, so
        // best-val is earlier than the final step.
        val finalModel = trainModel(
            tok, sampleCorpus(), steps = 600, lr = 5e-3, cfg = cfg, verbose = false,
            evalEvery = 100, saveBest = file.path,
        )
        assertTrue(file.exists(), "best checkpoint was not written")
        val (best, _) = Checkpoint.load(file.path)
        val bestVal = heldOutLoss(best, data, split, cfg.blockSize, 128)
        val finalVal = heldOutLoss(finalModel, data, split, cfg.blockSize, 128)
        assertTrue(bestVal <= finalVal + 1e-9, "saved best val $bestVal should be <= final $finalVal")
    }

    /** LR schedule ramps up during warmup then cosine-decays to minLr. */
    @Test
    fun scheduleWarmsUpThenCosineDecays() {
        val steps = 100; val warmup = 10; val peak = 1e-3; val minLr = 1e-4
        // Warmup: strictly increasing, ending at the peak.
        var prev = -1.0
        for (s in 0 until warmup) {
            val lr = scheduledLr(s, steps, peak, warmup, cosine = true, minLr = minLr)
            assertTrue(lr > prev, "warmup not increasing at $s"); prev = lr
        }
        assertEquals(peak, scheduledLr(warmup - 1, steps, peak, warmup, true, minLr), 1e-12)
        // After warmup: cosine peak -> min, non-increasing, within bounds.
        assertEquals(peak, scheduledLr(warmup, steps, peak, warmup, true, minLr), 1e-9)
        assertEquals(minLr, scheduledLr(steps - 1, steps, peak, warmup, true, minLr), peak * 1e-3)
        prev = Double.MAX_VALUE
        for (s in warmup until steps) {
            val lr = scheduledLr(s, steps, peak, warmup, true, minLr)
            assertTrue(lr <= prev + 1e-12 && lr in (minLr - 1e-9)..(peak + 1e-9)); prev = lr
        }
        // cosine=false holds the peak (after warmup).
        assertEquals(peak, scheduledLr(50, steps, peak, warmup, cosine = false, minLr = minLr), 1e-12)
    }

    /** Gradient clipping scales the global norm down to the cap, else leaves it. */
    @Test
    fun gradientClippingBoundsTheNorm() {
        fun norm(ps: List<Tensor>): Double {
            var s = 0.0; for (p in ps) for (g in p.grad) s += g * g; return Math.sqrt(s)
        }
        val a = Tensor(1, 3).also { it.grad[0] = 3.0; it.grad[1] = 4.0 }   // norm 5 with b
        val b = Tensor(1, 1).also { it.grad[0] = 0.0 }
        val ps = listOf(a, b)
        assertEquals(5.0, norm(ps), 1e-9)
        val returned = clipGradNorm(ps, maxNorm = 1.0)
        assertEquals(5.0, returned, 1e-9)                 // returns the pre-clip norm
        assertEquals(1.0, norm(ps), 1e-4)                 // scaled down to the cap
        // Below the cap: untouched.
        val c = Tensor(1, 2).also { it.grad[0] = 0.3; it.grad[1] = 0.4 }  // norm 0.5
        clipGradNorm(listOf(c), maxNorm = 10.0)
        assertEquals(0.5, norm(listOf(c)), 1e-9)
    }

    /** heldOutLoss must equal a direct eval-mode cross-entropy on its windows. */
    @Test
    fun heldOutLossMatchesDirectEval() {
        val tok = CharTokenizer(sampleCorpus())
        val data = tok.encode(sampleCorpus())
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = 16, nEmbed = 16, nHead = 2, nLayer = 1, dropout = 0.3)
        val model = GPT(cfg, seed = 3)
        val from = (data.size * 0.9).toInt()

        // With count=1 the single window is the region start; compare to a manual
        // eval-mode forward (train=false => dropout disabled, deterministic).
        val idx = data.copyOfRange(from, from + cfg.blockSize)
        val tgt = data.copyOfRange(from + 1, from + cfg.blockSize + 1)
        val direct = Tensor.crossEntropy(model.forward(idx), tgt).data[0]
        assertEquals(direct, heldOutLoss(model, data, from, cfg.blockSize, count = 1), 1e-12)
        // A held-out loss over many windows is finite and sane (< random baseline).
        val many = heldOutLoss(model, data, from, cfg.blockSize, count = 32)
        assertTrue(many.isFinite() && many < Math.log(cfg.vocabSize.toDouble()) * 2)
    }

    /** Dropout is a no-op at eval (training=false) and identity when p=0. */
    @Test
    fun dropoutIsIdentityAtEval() {
        val x = Tensor(2, 3).also { for (i in it.data.indices) it.data[i] = i + 1.0 }
        val evalOut = Tensor.dropout(x, 0.5, training = false, rng = Random(0))
        val zeroOut = Tensor.dropout(x, 0.0, training = true, rng = Random(0))
        for (i in x.data.indices) {
            assertEquals(x.data[i], evalOut.data[i])
            assertEquals(x.data[i], zeroOut.data[i])
        }
    }

    /** KV-cache decoding must reproduce the full forward pass's logits. */
    @Test
    fun kvCacheMatchesFullForward() {
        // Cover learned/RoPE positions and the tied output projection.
        val variants = listOf(Pair(false, false), Pair(true, false), Pair(false, true))
        for ((useRope, tie) in variants) {
            val tok = CharTokenizer(corpus())
            val cfg = Config(vocabSize = tok.vocabSize, blockSize = 12, nEmbed = 24, nHead = 3, nLayer = 2, useRope = useRope, tieWeights = tie)
            val model = trainModel(tok, corpus(), steps = 40, cfg = cfg, verbose = false)
            val ids = tok.encode(corpus()).copyOfRange(0, cfg.blockSize)

            val full = model.forward(ids)              // (T x vocab): row t = logits after token t
            val gen = KVGenerator(model)
            for (t in ids.indices) {
                val step = gen.step(ids[t])            // logits after feeding token t
                for (j in 0 until cfg.vocabSize) {
                    val diff = Math.abs(full.at(t, j) - step[j])
                    assertTrue(diff < 1e-6, "rope=$useRope tie=$tie mismatch at t=$t j=$j: $diff")
                }
            }
        }
    }

    /** A saved model must reload to bit-identical predictions. */
    @Test
    fun checkpointRoundTrips() {
        for (kind in listOf("char", "bpe")) {
            val text = corpus()
            val tok = makeTokenizer(kind, text)
            val cfg = Config(vocabSize = tok.vocabSize, blockSize = 8, nEmbed = 16, nHead = 2, nLayer = 1, useRope = kind == "bpe")
            val model = trainModel(tok, text, steps = 30, cfg = cfg, verbose = false)
            val ids = tok.encode(text).copyOfRange(0, cfg.blockSize)
            val before = model.forward(ids)

            val file = File.createTempFile("kortex-$kind", ".bin").apply { deleteOnExit() }
            Checkpoint.save(file.path, cfg, model, tok)
            val (loaded, loadedTok) = Checkpoint.load(file.path)

            // Tokenizer survives the round trip...
            assertEquals(text, loadedTok.decode(loadedTok.encode(text)), "$kind tokenizer changed after load")
            // ...and the reloaded weights produce identical logits.
            val after = loaded.forward(ids)
            for (i in before.data.indices) {
                assertTrue(Math.abs(before.data[i] - after.data[i]) < 1e-12, "$kind logits differ after reload at $i")
            }
        }
    }

    /** Backprop is correct if analytic gradients match numerical ones. */
    @Test
    fun backpropMatchesNumericalGradients() {
        val err = computeMaxGradError()
        assertTrue(err < 1e-4, "gradient check max relative error too high: $err")
    }

    @Test
    fun charTokenizerIsLossless() {
        val text = corpus()
        val tok = CharTokenizer(text)
        assertEquals(text, tok.decode(tok.encode(text)))
    }

    @Test
    fun bpeIsLossless() {
        val tok = BpeTokenizer.train(corpus(), targetVocab = 300)
        // Round-trips the training text...
        assertEquals(corpus(), tok.decode(tok.encode(corpus())))
        // ...and arbitrary unseen UTF-8 (byte-level => no "unknown token").
        val unseen = "Grüße! 42 — café; ∑x²  🙂"
        assertEquals(unseen, tok.decode(tok.encode(unseen)))
    }

    @Test
    fun bpeCompressesSequence() {
        val text = corpus()
        val tok = BpeTokenizer.train(text, targetVocab = 300)
        val nTokens = tok.encode(text).size
        assertTrue(nTokens < text.length, "BPE did not compress: $nTokens tokens vs ${text.length} chars")
    }

    @Test
    fun attentionIsCausalAndNormalized() {
        val tok = CharTokenizer(corpus())
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = 8, nEmbed = 16, nHead = 2, nLayer = 2)
        val model = GPT(cfg, seed = 1)
        val ids = tok.encode(corpus()).copyOfRange(0, cfg.blockSize)

        val sink = AttnSink()
        model.forward(ids, sink)
        assertEquals(cfg.nLayer * cfg.nHead, sink.records.size, "expected one matrix per (layer, head)")

        for ((layer, head, m) in sink.records) {
            for (i in ids.indices) {
                var rowSum = 0.0
                for (j in ids.indices) {
                    if (j > i) assertTrue(m[i][j] < 1e-6, "future leak at L$layer H$head [$i,$j]=${m[i][j]}")
                    rowSum += m[i][j]
                }
                assertTrue(Math.abs(rowSum - 1.0) < 1e-6, "row $i not normalized (sum=$rowSum) at L$layer H$head")
            }
        }
    }

    /** The RoPE rotation must backprop correctly too. */
    @Test
    fun backpropMatchesNumericalGradients_rope() {
        val err = computeMaxGradError(useRope = true)
        assertTrue(err < 1e-4, "RoPE gradient check max relative error too high: $err")
    }

    /** RoPE injects position by rotation, so it needs no learned position table. */
    @Test
    fun ropeDropsThePositionTable() {
        val learned = Config(vocabSize = 30, blockSize = 8, nEmbed = 16, nHead = 2, nLayer = 1)
        val rope = learned.copy(useRope = true)
        val pLearned = GPT(learned, seed = 1).parameters().sumOf { it.data.size }
        val pRope = GPT(rope, seed = 1).parameters().sumOf { it.data.size }
        assertEquals(learned.blockSize * learned.nEmbed, pLearned - pRope,
            "RoPE should save exactly the position-embedding table")
    }

    @Test
    fun trainingReducesLoss() {
        val text = corpus()
        val tok = CharTokenizer(text)
        val cfg = Config(vocabSize = tok.vocabSize, blockSize = 8, nEmbed = 16, nHead = 2, nLayer = 1)
        val data = tok.encode(text)
        val idx = data.copyOfRange(0, cfg.blockSize)
        val tgt = data.copyOfRange(1, cfg.blockSize + 1)

        // trainModel builds GPT(cfg, seed=1234); a fresh one with the same seed
        // has identical initial weights, giving a fair "before" baseline.
        val before = GPT(cfg, seed = 1234).loss(idx, tgt).data[0]
        val trained = trainModel(tok, text, steps = 150, cfg = cfg, verbose = false)
        val after = trained.loss(idx, tgt).data[0]

        assertTrue(after < before, "loss did not decrease: before=$before after=$after")
    }
}
