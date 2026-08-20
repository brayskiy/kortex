# Linear algebra in Kortex

A common misreading of the project's tagline: *"no linear-algebra libraries"* does
**not** mean *"no linear algebra."* Kortex is almost entirely linear algebra — we
just hand-write every operation in plain `Double` loops instead of importing BLAS,
ND4J, or a matrix library. This page maps each operation to where it lives and
what it's for.

The headline takeaway: **a from-scratch GPT bottoms out at essentially one
primitive — matrix multiply** — plus transpose, add, scale, and a 2×2 rotation.
Everything expensive is `matmul`; the heavyweight decompositions (inverse, SVD,
eigen) are exactly what gradient descent lets you skip.

## What we use, and for what

| Operation | Location | Purpose |
|-----------|----------|---------|
| **Matrix multiply** | [`Tensor.matmul`](../src/main/kotlin/Tensor.kt#L66) | The workhorse: Q/K/V projections (`x·Wq`), attention scores (`Q·Kᵀ`), attention output (`A·V`), head mixing (`concat·Wo`), both MLP layers, and final logits. |
| **Matrix–vector product** | [`KVGenerator.matVec`](../src/main/kotlin/Inference.kt#L147) | The KV-cache decode path handles one token at a time, so `matmul` degenerates to `Wᵀx` — cheaper, identical math. |
| **Transpose** | [`Tensor.transpose`](../src/main/kotlin/Tensor.kt#L120) | `Kᵀ` in attention scores; `tokEmbᵀ` when weights are tied to produce logits. |
| **Dot / inner product** | [`Inference.kt` attention](../src/main/kotlin/Inference.kt#L92) | The heart of attention — each query·key similarity, per head. |
| **Vector add + broadcasting** | [`Tensor.plus`](../src/main/kotlin/Tensor.kt#L92) | Residual connections (`x + sublayer(x)`) and bias rows broadcast over a batch. |
| **Scalar scaling** | [`Tensor.scale`](../src/main/kotlin/Tensor.kt#L110) | The `1/√d_k` factor applied to attention scores before softmax. |
| **2×2 rotation matrices** | [`Tensor.rope`](../src/main/kotlin/Tensor.kt#L309) | RoPE rotates each adjacent pair of dimensions by an angle ∝ position — literally applying `[[cosθ, −sinθ], [sinθ, cosθ]]`. Pure geometry. |
| **Outer-product accumulation** | inside [`matmul`'s backward](../src/main/kotlin/Tensor.kt#L66) | Backprop of a matmul is `dW = xᵀ·dOut` and `dA = dOut·Bᵀ` — the gradient is a sum of outer products. |
| **Mean / variance / standardization** | [`Tensor.layerNorm`](../src/main/kotlin/Tensor.kt#L216) | LayerNorm subtracts the row mean and divides by the row's standard deviation. |
| **L2 (Euclidean) norm** | [`clipGradNorm`](../src/main/kotlin/Train.kt#L107) | Gradient clipping computes the global gradient norm `‖g‖₂ = √Σg²` and rescales all gradients if it exceeds the cap. |
| **Row gather / selection** | [`Tensor.gatherRows`](../src/main/kotlin/Tensor.kt#L206) | Embedding lookup — selecting rows of the token / position matrices by index. |
| **Column slice & concat** | [`sliceCols`](../src/main/kotlin/Tensor.kt#L175) / [`concatCols`](../src/main/kotlin/Tensor.kt#L187) | Splitting the hidden vector into attention heads and rejoining them. |

### Not linear algebra (listed to avoid confusion)

These act *on* vectors but are nonlinear or probabilistic, so they don't count:
[`softmaxRows`](../src/main/kotlin/Tensor.kt#L130),
[`gelu`](../src/main/kotlin/Tensor.kt#L154),
[`crossEntropy`](../src/main/kotlin/Tensor.kt#L275),
[`causalMask`](../src/main/kotlin/Tensor.kt#L257),
[`dropout`](../src/main/kotlin/Tensor.kt#L342).

## What we *could* use but deliberately don't

The classical matrix routines — and it's instructive *why* a transformer needs
none of them:

- **Matrix inversion / solving `Ax = b`** — needed by **second-order optimizers**
  (Newton's method, natural gradient, K-FAC). Kortex uses **Adam**, which is
  first-order with a *diagonal* (per-element) preconditioner, so it never inverts
  or solves anything. This is the single biggest reason deep learning avoids heavy
  linear algebra.
- **Eigendecomposition / SVD** — would enable PCA of the learned embeddings, a
  **spectral norm** for tighter weight regularization, or analysis of what
  attention heads span. Useful for *inspection*, not for training.
- **QR decomposition** — the usual way to build an **orthogonal weight
  initialization**; Kortex uses scaled Gaussian init instead.
- **Cholesky / LU factorization** — for solving linear systems efficiently; only
  relevant if a second-order method were added.
- **Determinants** — appear in things like normalizing-flow likelihoods;
  irrelevant to a plain GPT.

## Why this matters

Reducing the whole network to `matmul` is exactly what makes LLMs trainable at
scale: one primitive that GPUs execute extremely fast, differentiated by a rule
(`dW = xᵀ·dOut`) that is itself just more matmuls. The decompositions above are
powerful but `O(n³)` and awkward to parallelize — gradient descent's trick is to
*approximate* their effect iteratively with first-order steps, so you never pay
for them.
