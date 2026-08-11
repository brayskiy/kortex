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

    /** KV-cache decoding must reproduce the full forward pass's logits. */
    @Test
    fun kvCacheMatchesFullForward() {
        for (useRope in listOf(false, true)) {
            val tok = CharTokenizer(corpus())
            val cfg = Config(vocabSize = tok.vocabSize, blockSize = 12, nEmbed = 24, nHead = 3, nLayer = 2, useRope = useRope)
            val model = trainModel(tok, corpus(), steps = 40, cfg = cfg, verbose = false)
            val ids = tok.encode(corpus()).copyOfRange(0, cfg.blockSize)

            val full = model.forward(ids)              // (T x vocab): row t = logits after token t
            val gen = KVGenerator(model)
            for (t in ids.indices) {
                val step = gen.step(ids[t])            // logits after feeding token t
                for (j in 0 until cfg.vocabSize) {
                    val diff = Math.abs(full.at(t, j) - step[j])
                    assertTrue(diff < 1e-6, "rope=$useRope mismatch at t=$t j=$j: $diff")
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
