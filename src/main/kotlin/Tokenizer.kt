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
}

/** One integer per distinct character in the training text. */
class CharTokenizer(text: String) : Tokenizer {
    val chars = text.toSortedSet().toList()
    override val vocabSize = chars.size
    private val stoi = chars.withIndex().associate { (i, c) -> c to i }
    private val itos = chars.withIndex().associate { (i, c) -> i to c }
    override fun encode(s: String): IntArray = IntArray(s.length) { stoi.getValue(s[it]) }
    override fun decode(ids: IntArray): String = ids.map { itos.getValue(it) }.joinToString("")
}

/**
 * Byte-level BPE. Ids 0..255 are the raw bytes; ids >= 256 are learned merges.
 * `vocab[id]` is the byte sequence a token expands to (for decoding), and
 * `newId[pair]` / `rank[pair]` drive greedy encoding.
 */
class BpeTokenizer private constructor(
    override val vocabSize: Int,
    private val vocab: Array<ByteArray>,     // id -> bytes it expands to
    private val newId: HashMap<Long, Int>,   // packed (a,b) -> merged id
    private val rank: HashMap<Long, Int>,    // packed (a,b) -> merge priority (lower = earlier)
) : Tokenizer {

    override fun encode(s: String): IntArray {
        val bytes = s.toByteArray(Charsets.UTF_8)
        if (bytes.isEmpty()) return IntArray(0)
        val ids = ArrayList<Int>(bytes.size)
        for (b in bytes) ids.add(b.toInt() and 0xff)
        // Greedily apply the lowest-rank merge available, all occurrences, repeat.
        while (ids.size >= 2) {
            var bestRank = Int.MAX_VALUE
            var bestPair = 0L
            for (i in 0 until ids.size - 1) {
                val p = pack(ids[i], ids[i + 1])
                val r = rank[p] ?: continue
                if (r < bestRank) { bestRank = r; bestPair = p }
            }
            if (bestRank == Int.MAX_VALUE) break   // no more learned merges apply
            val merged = newId.getValue(bestPair)
            val a = (bestPair ushr 32).toInt()
            val b = (bestPair and 0xffffffffL).toInt()
            val out = ArrayList<Int>(ids.size)
            var i = 0
            while (i < ids.size) {
                if (i < ids.size - 1 && ids[i] == a && ids[i + 1] == b) { out.add(merged); i += 2 }
                else { out.add(ids[i]); i += 1 }
            }
            ids.clear(); ids.addAll(out)
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

    companion object {
        /** Pack two ids into one Long key so we can use a HashMap for pairs. */
        private fun pack(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xffffffffL)

        /**
         * Learn merges from `text` until the vocabulary reaches `targetVocab`
         * (or no repeated pair remains). Educational: operates on the whole
         * corpus with no word-boundary pre-tokenization.
         */
        fun train(text: String, targetVocab: Int): BpeTokenizer {
            require(targetVocab >= 256) { "byte-level BPE needs vocab >= 256" }
            val vocab = ArrayList<ByteArray>(targetVocab)
            for (b in 0 until 256) vocab.add(byteArrayOf(b.toByte()))
            val newId = HashMap<Long, Int>()
            val rank = HashMap<Long, Int>()

            var ids = ArrayList<Int>()
            for (b in text.toByteArray(Charsets.UTF_8)) ids.add(b.toInt() and 0xff)

            while (vocab.size < targetVocab) {
                // Count every adjacent pair in the current sequence.
                val counts = HashMap<Long, Int>()
                for (i in 0 until ids.size - 1) {
                    val p = pack(ids[i], ids[i + 1])
                    counts[p] = (counts[p] ?: 0) + 1
                }
                // Most frequent pair; stop if nothing repeats.
                var best = 0L; var bestC = 1
                for ((p, c) in counts) if (c > bestC) { bestC = c; best = p }
                if (bestC <= 1) break

                val a = (best ushr 32).toInt()
                val b = (best and 0xffffffffL).toInt()
                val id = vocab.size
                vocab.add(vocab[a] + vocab[b])
                newId[best] = id
                rank[best] = rank.size

                // Replace every occurrence of the pair with the new id.
                val out = ArrayList<Int>(ids.size)
                var i = 0
                while (i < ids.size) {
                    if (i < ids.size - 1 && ids[i] == a && ids[i + 1] == b) { out.add(id); i += 2 }
                    else { out.add(ids[i]); i += 1 }
                }
                ids = out
            }
            return BpeTokenizer(vocab.size, vocab.toTypedArray(), newId, rank)
        }
    }
}
