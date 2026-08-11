/*
 * Tensor.kt — a tiny reverse-mode autograd engine over 2D matrices.
 *
 * This is the "learning machine" underneath every neural network. A Tensor holds
 * numbers (`data`) and, after a backward pass, the gradient of the loss with
 * respect to each of those numbers (`grad`). Every operation (matmul, softmax,
 * ...) records how to push gradients back to its inputs (`backwardFn`). Calling
 * `.backward()` on the final scalar loss runs those closures in reverse
 * topological order — this IS backpropagation.
 *
 * Everything an LLM does at train time is: build a graph of these ops, compute a
 * loss, call backward(), then nudge the parameters against their gradients.
 */

import java.util.Random
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.math.PI

/** A 2D matrix (rows x cols) that can track gradients through a compute graph. */
class Tensor(
    val rows: Int,
    val cols: Int,
    val data: DoubleArray = DoubleArray(rows * cols),
) {
    /** dLoss/dData, filled in during backward(). Same shape as `data`. */
    val grad: DoubleArray = DoubleArray(rows * cols)

    /** Inputs this tensor was computed from (empty for leaves/parameters). */
    private var parents: List<Tensor> = emptyList()

    /** How to add this node's contribution to its parents' gradients. */
    private var backwardFn: (() -> Unit)? = null

    fun at(r: Int, c: Int) = data[r * cols + c]

    private fun build(parents: List<Tensor>, backward: () -> Unit): Tensor {
        this.parents = parents
        this.backwardFn = backward
        return this
    }

    /** Run backprop from this (assumed scalar) tensor through the whole graph. */
    fun backward() {
        // 1) Topological order of all nodes reachable from here.
        val topo = ArrayList<Tensor>()
        val seen = HashSet<Tensor>()
        fun visit(t: Tensor) {
            if (t in seen) return
            seen.add(t)
            for (p in t.parents) visit(p)
            topo.add(t)
        }
        visit(this)
        // 2) Seed: dLoss/dLoss = 1.
        grad.fill(0.0); grad[0] = 1.0
        // 3) Walk backwards, letting each node push grads to its parents.
        for (i in topo.indices.reversed()) topo[i].backwardFn?.invoke()
    }

    // ---- Operations. Each returns a NEW tensor and wires up its backward. ----

    /** Matrix multiply: (m x k) @ (k x n) = (m x n). */
    infix fun matmul(b: Tensor): Tensor {
        val a = this
        require(a.cols == b.rows) { "matmul shape ${a.rows}x${a.cols} @ ${b.rows}x${b.cols}" }
        val m = a.rows; val k = a.cols; val n = b.cols
        val out = Tensor(m, n)
        for (i in 0 until m) for (j in 0 until n) {
            var s = 0.0
            for (p in 0 until k) s += a.data[i * k + p] * b.data[p * n + j]
            out.data[i * n + j] = s
        }
        return out.build(listOf(a, b)) {
            // dA += dOut @ B^T ; dB += A^T @ dOut
            for (i in 0 until m) for (p in 0 until k) {
                var s = 0.0
                for (j in 0 until n) s += out.grad[i * n + j] * b.data[p * n + j]
                a.grad[i * k + p] += s
            }
            for (p in 0 until k) for (j in 0 until n) {
                var s = 0.0
                for (i in 0 until m) s += a.data[i * k + p] * out.grad[i * n + j]
                b.grad[p * n + j] += s
            }
        }
    }

    /** Add. Same shape, OR add a bias row (1 x cols) broadcast over every row. */
    operator fun plus(b: Tensor): Tensor {
        val a = this
        val broadcast = b.rows == 1 && a.rows != 1
        require(a.cols == b.cols && (broadcast || a.rows == b.rows)) { "add shapes" }
        val out = Tensor(a.rows, a.cols)
        for (i in 0 until a.rows) for (j in 0 until a.cols) {
            out.data[i * a.cols + j] = a.data[i * a.cols + j] + b.data[(if (broadcast) 0 else i) * a.cols + j]
        }
        return out.build(listOf(a, b)) {
            for (i in 0 until a.rows) for (j in 0 until a.cols) {
                val g = out.grad[i * a.cols + j]
                a.grad[i * a.cols + j] += g
                b.grad[(if (broadcast) 0 else i) * a.cols + j] += g
            }
        }
    }

    /** Multiply every element by a constant scalar. */
    fun scale(s: Double): Tensor {
        val a = this
        val out = Tensor(rows, cols)
        for (i in data.indices) out.data[i] = a.data[i] * s
        return out.build(listOf(a)) {
            for (i in data.indices) a.grad[i] += out.grad[i] * s
        }
    }

    /** Transpose (rows x cols) -> (cols x rows). */
    fun transpose(): Tensor {
        val a = this
        val out = Tensor(cols, rows)
        for (i in 0 until rows) for (j in 0 until cols) out.data[j * rows + i] = a.data[i * cols + j]
        return out.build(listOf(a)) {
            for (i in 0 until rows) for (j in 0 until cols) a.grad[i * cols + j] += out.grad[j * rows + i]
        }
    }

    /** Row-wise softmax: each row becomes a probability distribution. */
    fun softmaxRows(): Tensor {
        val a = this
        val out = Tensor(rows, cols)
        for (i in 0 until rows) {
            var mx = Double.NEGATIVE_INFINITY
            for (j in 0 until cols) mx = maxOf(mx, a.data[i * cols + j])
            var sum = 0.0
            for (j in 0 until cols) { val e = exp(a.data[i * cols + j] - mx); out.data[i * cols + j] = e; sum += e }
            for (j in 0 until cols) out.data[i * cols + j] /= sum
        }
        return out.build(listOf(a)) {
            // dx_i = y_i * (dy_i - sum_j dy_j y_j)
            for (i in 0 until rows) {
                var dot = 0.0
                for (j in 0 until cols) dot += out.grad[i * cols + j] * out.data[i * cols + j]
                for (j in 0 until cols) {
                    val y = out.data[i * cols + j]
                    a.grad[i * cols + j] += y * (out.grad[i * cols + j] - dot)
                }
            }
        }
    }

    /** GELU nonlinearity (tanh approximation), applied elementwise. */
    fun gelu(): Tensor {
        val a = this
        val c = sqrt(2.0 / PI)
        val out = Tensor(rows, cols)
        for (i in data.indices) {
            val x = a.data[i]
            out.data[i] = 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)))
        }
        return out.build(listOf(a)) {
            for (i in data.indices) {
                val x = a.data[i]
                val u = c * (x + 0.044715 * x * x * x)
                val t = tanh(u)
                val dudx = c * (1.0 + 3.0 * 0.044715 * x * x)
                val d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dudx
                a.grad[i] += out.grad[i] * d
            }
        }
    }

    /** Slice a contiguous block of columns [c0, c1). Used to split attention heads. */
    fun sliceCols(c0: Int, c1: Int): Tensor {
        val a = this
        val w = c1 - c0
        val out = Tensor(rows, w)
        for (i in 0 until rows) for (j in 0 until w) out.data[i * w + j] = a.data[i * cols + (c0 + j)]
        return out.build(listOf(a)) {
            for (i in 0 until rows) for (j in 0 until w) a.grad[i * cols + (c0 + j)] += out.grad[i * w + j]
        }
    }

    companion object {
        /** Concatenate tensors side by side (same rows). Rejoins attention heads. */
        fun concatCols(parts: List<Tensor>): Tensor {
            val rows = parts[0].rows
            val totalCols = parts.sumOf { it.cols }
            val out = Tensor(rows, totalCols)
            var off = 0
            for (p in parts) {
                for (i in 0 until rows) for (j in 0 until p.cols) out.data[i * totalCols + (off + j)] = p.data[i * p.cols + j]
                off += p.cols
            }
            return out.build(parts) {
                var o = 0
                for (p in parts) {
                    for (i in 0 until rows) for (j in 0 until p.cols) p.grad[i * p.cols + j] += out.grad[i * totalCols + (o + j)]
                    o += p.cols
                }
            }
        }

        /** Gather rows of `table` by integer indices -> (indices.size x table.cols). */
        fun gatherRows(table: Tensor, idx: IntArray): Tensor {
            val d = table.cols
            val out = Tensor(idx.size, d)
            for (i in idx.indices) for (j in 0 until d) out.data[i * d + j] = table.data[idx[i] * d + j]
            return out.build(listOf(table)) {
                for (i in idx.indices) for (j in 0 until d) table.grad[idx[i] * d + j] += out.grad[i * d + j]
            }
        }

        /** Row-wise LayerNorm with learnable gamma/beta (both shape 1 x d). */
        fun layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: Double = 1e-5): Tensor {
            val n = x.cols
            val out = Tensor(x.rows, n)
            val xhat = DoubleArray(x.rows * n)
            val invStd = DoubleArray(x.rows)
            for (i in 0 until x.rows) {
                var mean = 0.0
                for (j in 0 until n) mean += x.data[i * n + j]
                mean /= n
                var v = 0.0
                for (j in 0 until n) { val d = x.data[i * n + j] - mean; v += d * d }
                v /= n
                val inv = 1.0 / sqrt(v + eps)
                invStd[i] = inv
                for (j in 0 until n) {
                    val h = (x.data[i * n + j] - mean) * inv
                    xhat[i * n + j] = h
                    out.data[i * n + j] = gamma.data[j] * h + beta.data[j]
                }
            }
            return out.build(listOf(x, gamma, beta)) {
                for (i in 0 until x.rows) {
                    var meanDh = 0.0
                    var meanDhXhat = 0.0
                    for (j in 0 until n) {
                        val dh = out.grad[i * n + j] * gamma.data[j]
                        meanDh += dh
                        meanDhXhat += dh * xhat[i * n + j]
                    }
                    meanDh /= n; meanDhXhat /= n
                    for (j in 0 until n) {
                        val dh = out.grad[i * n + j] * gamma.data[j]
                        x.grad[i * n + j] += invStd[i] * (dh - meanDh - xhat[i * n + j] * meanDhXhat)
                        gamma.grad[j] += out.grad[i * n + j] * xhat[i * n + j]
                        beta.grad[j] += out.grad[i * n + j]
                    }
                }
            }
        }

        /** Add a causal mask: position i may not attend to future positions j > i. */
        fun causalMask(scores: Tensor): Tensor {
            val n = scores.cols
            val out = Tensor(scores.rows, n)
            for (i in 0 until scores.rows) for (j in 0 until n) {
                out.data[i * n + j] = if (j > i) -1e9 else scores.data[i * n + j]
            }
            return out.build(listOf(scores)) {
                for (i in 0 until scores.rows) for (j in 0 until n) {
                    if (j <= i) scores.grad[i * n + j] += out.grad[i * n + j]
                }
            }
        }

        /**
         * Cross-entropy loss for next-token prediction.
         * logits: (T x vocab), targets: T integer labels. Returns scalar loss.
         * This is the single number the whole network is trained to minimize.
         */
        fun crossEntropy(logits: Tensor, targets: IntArray): Tensor {
            val t = logits.rows; val v = logits.cols
            val probs = DoubleArray(t * v)
            var loss = 0.0
            for (i in 0 until t) {
                var mx = Double.NEGATIVE_INFINITY
                for (j in 0 until v) mx = maxOf(mx, logits.data[i * v + j])
                var sum = 0.0
                for (j in 0 until v) { val e = exp(logits.data[i * v + j] - mx); probs[i * v + j] = e; sum += e }
                for (j in 0 until v) probs[i * v + j] /= sum
                loss += -ln(probs[i * v + targets[i]] + 1e-12)
            }
            loss /= t
            val out = Tensor(1, 1, doubleArrayOf(loss))
            return out.build(listOf(logits)) {
                val g = out.grad[0] / t
                for (i in 0 until t) for (j in 0 until v) {
                    val soft = probs[i * v + j] - if (j == targets[i]) 1.0 else 0.0
                    logits.grad[i * v + j] += g * soft
                }
            }
        }

        /**
         * Rotary Positional Embedding (RoPE). Rotates each adjacent pair of
         * dimensions of every row by an angle proportional to the row's position.
         * `x` is (T x headDim), row i = token at position i. Applied to queries
         * and keys inside attention, this makes the dot-product Q·K depend only on
         * the *relative* offset between two tokens — no learned position table.
         *
         * For pair k at position i with angle θ = i · base^(-2k/d):
         *   [x_a, x_b] -> [x_a·cosθ - x_b·sinθ,  x_a·sinθ + x_b·cosθ]   (a 2D rotation)
         * The backward is the transposed (inverse) rotation.
         */
        fun rope(x: Tensor, base: Double = 10000.0): Tensor {
            val t = x.rows; val d = x.cols
            require(d % 2 == 0) { "RoPE needs an even head dim, got $d" }
            val half = d / 2
            val cos = DoubleArray(t * half); val sin = DoubleArray(t * half)
            for (i in 0 until t) for (k in 0 until half) {
                val ang = i * Math.pow(base, -2.0 * k / d)
                cos[i * half + k] = Math.cos(ang)
                sin[i * half + k] = Math.sin(ang)
            }
            val out = Tensor(t, d)
            for (i in 0 until t) for (k in 0 until half) {
                val a = x.data[i * d + 2 * k]; val b = x.data[i * d + 2 * k + 1]
                val c = cos[i * half + k]; val s = sin[i * half + k]
                out.data[i * d + 2 * k] = a * c - b * s
                out.data[i * d + 2 * k + 1] = a * s + b * c
            }
            return out.build(listOf(x)) {
                for (i in 0 until t) for (k in 0 until half) {
                    val ga = out.grad[i * d + 2 * k]; val gb = out.grad[i * d + 2 * k + 1]
                    val c = cos[i * half + k]; val s = sin[i * half + k]
                    x.grad[i * d + 2 * k] += ga * c + gb * s
                    x.grad[i * d + 2 * k + 1] += -ga * s + gb * c
                }
            }
        }

        /** A parameter (leaf) tensor initialized with small random values. */
        fun param(rows: Int, cols: Int, rng: Random, std: Double = 0.02): Tensor {
            val t = Tensor(rows, cols)
            for (i in t.data.indices) t.data[i] = rng.nextGaussian() * std
            return t
        }

        fun zeros(rows: Int, cols: Int) = Tensor(rows, cols)
    }
}
