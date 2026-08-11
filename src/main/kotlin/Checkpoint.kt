/*
 * Checkpoint.kt — save and load a trained model to a single file.
 *
 * A checkpoint bundles everything needed to use a model later: the Config, the
 * tokenizer (its vocab / merges), and every parameter tensor's values. On load
 * we rebuild the GPT from the Config and overwrite its random weights with the
 * saved ones — the parameter order from GPT.parameters() is deterministic, so
 * save and load line up.
 */

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

object Checkpoint {
    private const val MAGIC = 0x4B525458   // "KRTX"
    private const val VERSION = 2

    fun save(path: String, cfg: Config, model: GPT, tok: Tokenizer) {
        DataOutputStream(BufferedOutputStream(File(path).outputStream())).use { o ->
            o.writeInt(MAGIC); o.writeInt(VERSION)
            o.writeInt(cfg.vocabSize); o.writeInt(cfg.blockSize)
            o.writeInt(cfg.nEmbed); o.writeInt(cfg.nHead); o.writeInt(cfg.nLayer)
            o.writeBoolean(cfg.useRope)
            o.writeBoolean(cfg.tieWeights); o.writeDouble(cfg.dropout)
            tok.save(o)
            val ps = model.parameters()
            o.writeInt(ps.size)
            for (p in ps) {
                o.writeInt(p.rows); o.writeInt(p.cols)
                for (d in p.data) o.writeDouble(d)
            }
        }
    }

    /** Returns the reconstructed model and its tokenizer. */
    fun load(path: String): Pair<GPT, Tokenizer> {
        DataInputStream(BufferedInputStream(File(path).inputStream())).use { i ->
            require(i.readInt() == MAGIC) { "$path is not a Kortex checkpoint" }
            val version = i.readInt()
            require(version == VERSION) { "unsupported checkpoint version $version" }
            val cfg = Config(
                vocabSize = i.readInt(), blockSize = i.readInt(),
                nEmbed = i.readInt(), nHead = i.readInt(), nLayer = i.readInt(),
                useRope = i.readBoolean(),
                tieWeights = i.readBoolean(), dropout = i.readDouble(),
            )
            val tok = loadTokenizer(i)
            val model = GPT(cfg)
            val ps = model.parameters()
            val n = i.readInt()
            require(n == ps.size) { "checkpoint has $n params, model expects ${ps.size}" }
            for (p in ps) {
                val r = i.readInt(); val c = i.readInt()
                require(r == p.rows && c == p.cols) { "param shape mismatch: $r x $c vs ${p.rows} x ${p.cols}" }
                for (k in p.data.indices) p.data[k] = i.readDouble()
            }
            return model to tok
        }
    }
}
