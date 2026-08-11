# Architecture

Kortex is a decoder-only transformer (a tiny GPT) with a hand-written
reverse-mode autograd engine. No ML libraries. This note explains how the pieces
fit together and where to look for each idea.

## Module map

```
src/main/kotlin/
  Tensor.kt      autograd engine: Tensor + every differentiable op
  Tokenizer.kt   Tokenizer interface, CharTokenizer, BpeTokenizer
  Model.kt       Config, Attention, MLP, Block, GPT, AttnSink
  Train.kt       Adam, gradient check, training loop, sampling, CLI
  Viz.kt         attention-weight visualizer (ASCII + HTML)
src/test/kotlin/
  KortexTest.kt  JUnit checks of the core invariants
```

## Data flow (one forward pass)

```
ids: IntArray (length T)
      │  gatherRows(tokEmb, ids)                       (T x d)
      │  + gatherRows(posEmb, 0..T-1)   ← learned positions only (skipped for RoPE)
      ▼
   x: (T x d)
      │
      ├─►┌─ Block ────────────────────────────────────────────────┐   × nLayer
      │  │ x = x + Attention(LayerNorm(x))   ← tokens mix here     │
      │  │ x = x + MLP(LayerNorm(x))         ← per-token compute   │
      │  └─────────────────────────────────────────────────────────┘
      ▼
   LayerNorm(x)                                        (T x d)
      │  matmul(head)
      ▼
 logits: (T x vocab)  ──► crossEntropy(logits, targets) ──► scalar loss
```

`loss.backward()` walks the graph in reverse topological order, and each op adds
its contribution to its inputs' `.grad`. `Adam.step()` then nudges every
parameter against its gradient.

### Inside Attention (`Attention.forward`)

```
x ─► Q = x·Wq, K = x·Wk, V = x·Wv           (each T x d)
   for each head h (a d/nHead-wide slice of Q,K,V):
      Qh, Kh, Vh                             (T x headDim)
      if RoPE: Qh = rope(Qh); Kh = rope(Kh)  ← position enters here
      scores = (Qh · Khᵀ) / √headDim         (T x T)
      scores = causalMask(scores)            ← upper triangle → −∞
      A = softmaxRows(scores)                ← rows sum to 1 (captured by AttnSink)
      head_out = A · Vh                       (T x headDim)
   concat heads ─► · Wo                       (T x d)
```

## The autograd engine (`Tensor.kt`)

A `Tensor` is a flat `DoubleArray data` with `rows × cols`, a matching `grad`
array, a list of `parents`, and a `backwardFn` closure. Every op:

1. computes the forward result into a new `Tensor`, then
2. calls `out.build(parents) { ... }` to register a closure that reads `out.grad`
   and **adds** into each parent's `grad`.

`backward()` seeds `dLoss/dLoss = 1` and invokes the closures in reverse topo
order. Because closures *accumulate* (`+=`), a tensor used in several places gets
the sum of all downstream gradients — exactly the multivariable chain rule.

Ops implemented: `matmul`, `plus` (with row-broadcast bias), `scale`,
`transpose`, `softmaxRows`, `gelu`, `layerNorm`, `sliceCols`, `concatCols`,
`gatherRows`, `causalMask`, `rope`, `crossEntropy`.

## Shapes and hyperparameters (`Config`)

| Symbol | Meaning | Where |
|--------|---------|-------|
| `vocabSize` | number of distinct tokens | tokenizer |
| `blockSize` | max context length `T` | positions |
| `nEmbed` (`d`) | hidden width | everywhere |
| `nHead` | attention heads (`headDim = d/nHead`) | attention |
| `nLayer` | number of `Block`s | model depth |
| `useRope` | rotary positions vs. learned table | positions |

Everything processes **one sequence at a time**; a "batch" is a loop that
accumulates gradients before a single optimizer step (`Train.kt`).

## Deliberate simplifications

Chosen for readability, not performance or SOTA quality:

- Scalar `Double` math in plain loops — no vectorization, no GPU, no threading.
- One sequence per forward pass (no batched tensor dimension).
- BPE has no word-boundary pre-tokenization (whole-corpus merges).
- Positions for a window are `0..T-1`. Training re-encodes the full context each
  step; generation can use the KV-cache (`Inference.kt`) for O(t)-per-step decoding
  within one context window.
- Weights are not tied between `tokEmb` and `head`.

## How to trust changes

`computeMaxGradError()` compares analytic gradients to finite differences; the
tests assert it stays below `1e-4` for both the learned-position and RoPE models.
Any new differentiable op should be validated the same way (see CONTRIBUTING.md).
