/*
 * Tokenizer.kt — turning text into integer ids and back.
 *
 * Two implementations behind one interface:
 *   CharTokenizer — one id per character. Trivial, tiny vocab.
 *   BpeTokenizer  — byte-level Byte-Pair Encoding, the algorithm GPT-2/GPT-3 use.
 *
 * BPE idea: start from raw bytes, then repeatedly find the most frequent adjacent
 * pair of symbols and merge it into a new symbol. Common sequences ("the", " is")
 * become single tokens, so text turns into FEWER tokens than characters — the
 * model sees longer context for the same sequence length. Working on raw *bytes*
 * (0..255) means any UTF-8 text encodes without an "unknown token".
 */

interface Tokenizer {
    val vocabSize: Int
    fun encode(s: String): IntArray
    fun decode(ids: IntArray): String
    /** Serialize enough state to reconstruct this tokenizer (see loadTokenizer). */
    fun save(out: java.io.DataOutputStream)
}

/** Rebuild a tokenizer written by Tokenizer.save. */
fun loadTokenizer(inp: java.io.DataInputStream): Tokenizer = when (val type = inp.readUTF()) {
    "char" -> CharTokenizer(inp.readUTF())          // the char string reproduces the sorted vocab
    "bpe" -> BpeTokenizer.read(inp)
    else -> error("unknown tokenizer type: $type")
}

/** One integer per distinct character in the training text. */
class CharTokenizer(text: String) : Tokenizer {
    val chars = text.toSortedSet().toList()
    override val vocabSize = chars.size
    private val stoi = chars.withIndex().associate { (i, c) -> c to i }
    private val itos = chars.withIndex().associate { (i, c) -> i to c }
    override fun encode(s: String): IntArray = IntArray(s.length) { stoi.getValue(s[it]) }
    override fun decode(ids: IntArray): String = ids.map { itos.getValue(it) }.joinToString("")
    override fun save(out: java.io.DataOutputStream) {
        out.writeUTF("char")
        out.writeUTF(chars.joinToString(""))
    }
}

/**
 * Byte-level BPE. Ids 0..255 are the raw bytes; ids >= 256 are learned merges.
 * `vocab[id]` is the byte sequence a token expands to (for decoding).
 *
 * Scales to larger files by pre-tokenizing into "words" (runs of whitespace or
 * non-whitespace via [WORD]) and training on the *frequency table of distinct
 * words* rather than the raw byte stream — the same trick GPT-2 uses. Merges
 * never cross a word boundary, and each distinct word is encoded once and cached,
 * so a big repetitive corpus costs work proportional to its vocabulary, not its
 * length.
 */
class BpeTokenizer private constructor(
    override val vocabSize: Int,
    private val vocab: Array<ByteArray>,     // id -> bytes it expands to
    private val newId: HashMap<Long, Int>,   // packed (a,b) -> merged id
    private val rank: HashMap<Long, Int>,    // packed (a,b) -> merge priority (lower = earlier)
) : Tokenizer {

    private val cache = HashMap<String, IntArray>()   // word -> its token ids

    override fun encode(s: String): IntArray {
        if (s.isEmpty()) return IntArray(0)
        val out = ArrayList<Int>(s.length)
        for (m in WORD.findAll(s)) {                  // tiles s exactly (lossless)
            val word = m.value
            val ids = cache.getOrPut(word) { encodeWord(word) }
            for (id in ids) out.add(id)
        }
        return out.toIntArray()
    }

    /** Apply merges within a single word: repeatedly merge the lowest-rank pair. */
    private fun encodeWord(word: String): IntArray {
        val ids = ArrayList<Int>()
        for (b in word.toByteArray(Charsets.UTF_8)) ids.add(b.toInt() and 0xff)
        while (ids.size >= 2) {
            var bestRank = Int.MAX_VALUE
            var bestPair = 0L
            for (i in 0 until ids.size - 1) {
                val p = pack(ids[i], ids[i + 1])
                val r = rank[p] ?: continue
                if (r < bestRank) { bestRank = r; bestPair = p }
            }
            if (bestRank == Int.MAX_VALUE) break
            val merged = newId.getValue(bestPair)
            val a = (bestPair ushr 32).toInt(); val b = (bestPair and 0xffffffffL).toInt()
            var i = 0
            val next = ArrayList<Int>(ids.size)
            while (i < ids.size) {
                if (i < ids.size - 1 && ids[i] == a && ids[i + 1] == b) { next.add(merged); i += 2 }
                else { next.add(ids[i]); i += 1 }
            }
            ids.clear(); ids.addAll(next)
        }
        return ids.toIntArray()
    }

    override fun decode(ids: IntArray): String {
        var total = 0
        for (id in ids) total += vocab[id].size
        val bytes = ByteArray(total)
        var off = 0
        for (id in ids) { val v = vocab[id]; System.arraycopy(v, 0, bytes, off, v.size); off += v.size }
        return String(bytes, Charsets.UTF_8)
    }

    override fun save(out: java.io.DataOutputStream) {
        out.writeUTF("bpe")
        // Only the merge pairs, in learned order, are needed — the vocab (byte
        // expansions) is rebuilt from them on load. Merged id == 256 + order.
        val ordered = newId.entries.sortedBy { it.value }
        out.writeInt(ordered.size)
        for (e in ordered) {
            out.writeInt((e.key ushr 32).toInt())          // a
            out.writeInt((e.key and 0xffffffffL).toInt())  // b
        }
    }

    companion object {
        /** Whitespace-run or non-whitespace-run; the alternation tiles any string. */
        private val WORD = Regex("\\s+|\\S+")

        /** Pack two ids into one Long key so we can use a HashMap for pairs. */
        private fun pack(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xffffffffL)

        /** Reconstruct from the merge list written by save(). */
        fun read(inp: java.io.DataInputStream): BpeTokenizer {
            val vocab = ArrayList<ByteArray>()
            for (b in 0 until 256) vocab.add(byteArrayOf(b.toByte()))
            val newId = HashMap<Long, Int>()
            val rank = HashMap<Long, Int>()
            val n = inp.readInt()
            for (k in 0 until n) {
                val a = inp.readInt(); val b = inp.readInt()
                val key = pack(a, b)
                newId[key] = vocab.size
                rank[key] = rank.size
                vocab.add(vocab[a] + vocab[b])
            }
            return BpeTokenizer(vocab.size, vocab.toTypedArray(), newId, rank)
        }

        /**
         * Learn merges until the vocabulary reaches `targetVocab` (or no repeated
         * pair remains). Trains on the distinct-word frequency table, so cost
         * scales with the number of unique words, not the corpus length.
         */
        fun train(text: String, targetVocab: Int): BpeTokenizer {
            require(targetVocab >= 256) { "byte-level BPE needs vocab >= 256" }
            val vocab = ArrayList<ByteArray>(targetVocab)
            for (b in 0 until 256) vocab.add(byteArrayOf(b.toByte()))
            val newId = HashMap<Long, Int>()
            val rank = HashMap<Long, Int>()

            // Distinct words -> frequency, each represented as a list of byte ids.
            val freq = HashMap<String, Int>()
            for (m in WORD.findAll(text)) freq[m.value] = (freq[m.value] ?: 0) + 1
            val words = ArrayList<ArrayList<Int>>(freq.size)
            val counts = IntArray(freq.size)
            var wi = 0
            for ((w, c) in freq) {
                val ids = ArrayList<Int>()
                for (b in w.toByteArray(Charsets.UTF_8)) ids.add(b.toInt() and 0xff)
                words.add(ids); counts[wi] = c; wi++
            }

            while (vocab.size < targetVocab) {
                // Count adjacent pairs across all words, weighted by word frequency.
                val pairCount = HashMap<Long, Int>()
                for (w in words.indices) {
                    val ids = words[w]; val c = counts[w]
                    for (i in 0 until ids.size - 1) {
                        val p = pack(ids[i], ids[i + 1])
                        pairCount[p] = (pairCount[p] ?: 0) + c
                    }
                }
                var best = 0L; var bestC = 1
                for ((p, c) in pairCount) if (c > bestC) { bestC = c; best = p }
                if (bestC <= 1) break

                val a = (best ushr 32).toInt(); val b = (best and 0xffffffffL).toInt()
                val id = vocab.size
                vocab.add(vocab[a] + vocab[b])
                newId[best] = id
                rank[best] = rank.size

                // Apply the merge inside every word.
                for (w in words.indices) {
                    val ids = words[w]
                    if (ids.size < 2) continue
                    var i = 0
                    val out = ArrayList<Int>(ids.size)
                    while (i < ids.size) {
                        if (i < ids.size - 1 && ids[i] == a && ids[i + 1] == b) { out.add(id); i += 2 }
                        else { out.add(ids[i]); i += 1 }
                    }
                    words[w] = out
                }
            }
            return BpeTokenizer(vocab.size, vocab.toTypedArray(), newId, rank)
        }
    }
}
