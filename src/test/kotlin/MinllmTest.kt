/*
 * MinllmTest.kt — automated checks run by `./gradlew test`.
 *
 * These assert the properties the whole thing depends on:
 *   - backprop matches finite-difference gradients (the math is right)
 *   - tokenizers are lossless (decode ∘ encode == identity)
 *   - BPE actually compresses the sequence
 *   - attention is causal and each row is a probability distribution
 *   - training reduces the loss
 */

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MinllmTest {

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
