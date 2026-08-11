/*
 * Sampling.kt — how the next token is chosen from the model's logits.
 *
 * The model outputs a score (logit) per vocabulary token; sampling turns those
 * into an actual choice. Three knobs shape the randomness:
 *
 *   temperature  scales logits before softmax. →0 sharpens (greedy, repetitive),
 *                >1 flattens (more surprising, more mistakes).
 *   top-k        keep only the k most likely tokens, then sample among them.
 *   top-p        (nucleus) keep the most likely tokens whose probabilities sum to
 *                at least p, then sample among them. Adapts the cutoff to how
 *                confident the model is.
 *
 * top-k and top-p compose: top-k is applied first, then top-p, then we renormalize
 * over whatever survives and draw one token.
 */

import java.util.Random
import kotlin.math.exp

data class Sampler(
    val temperature: Double = 0.8,
    val topK: Int = 0,        // 0 = disabled
    val topP: Double = 1.0,   // 1.0 = disabled
)

fun argmax(logits: DoubleArray): Int {
    var best = 0
    for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
    return best
}

/** Pick one token id from `logits` under the sampler's temperature/top-k/top-p. */
fun sampleFrom(logits: DoubleArray, s: Sampler, rng: Random): Int {
    val n = logits.size
    if (s.temperature <= 0.0) return argmax(logits)   // greedy

    // Temperature-scaled softmax -> probabilities.
    var mx = Double.NEGATIVE_INFINITY
    for (x in logits) if (x > mx) mx = x
    val p = DoubleArray(n)
    var sum = 0.0
    for (i in 0 until n) { val e = exp((logits[i] - mx) / s.temperature); p[i] = e; sum += e }
    for (i in 0 until n) p[i] /= sum

    // Token ids sorted by probability, most likely first.
    val order = (0 until n).sortedByDescending { p[it] }
    val keep = BooleanArray(n)

    // top-k: keep the k most likely.
    if (s.topK in 1 until n) for (r in 0 until s.topK) keep[order[r]] = true else keep.fill(true)

    // top-p: within the kept set, keep the smallest prefix reaching cumulative p.
    if (s.topP < 1.0) {
        val nucleus = BooleanArray(n)
        var cum = 0.0
        for (i in order) {
            if (!keep[i]) continue
            nucleus[i] = true
            cum += p[i]
            if (cum >= s.topP) break
        }
        for (i in 0 until n) keep[i] = keep[i] && nucleus[i]
    }

    // Renormalize over survivors and draw.
    var z = 0.0
    for (i in 0 until n) if (keep[i]) z += p[i]
    var r = rng.nextDouble() * z
    for (i in order) {
        if (!keep[i]) continue
        r -= p[i]
        if (r <= 0.0) return i
    }
    return order.first { keep[it] }
}
