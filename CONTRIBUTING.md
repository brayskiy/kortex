# Contributing to Kortex

Kortex is a **learning** project: an LLM small and clear enough to read end to
end. Contributions are welcome as long as they keep it that way. When clarity and
performance conflict, choose clarity.

## Prerequisites

- A JDK (17 or newer; developed on 21). Nothing else — the Gradle **wrapper**
  (`./gradlew`) fetches Gradle and Kotlin on first use.

## Build, test, run

```bash
./gradlew test                       # run the JUnit suite (do this before every PR)
./gradlew run --args="gradcheck"     # verify backprop (learned + RoPE)
./gradlew run --args="train char"    # <mode> [char|bpe] [learned|rope]
```

Run modes and their args are documented in the [README](README.md); the overall
design is in [ARCHITECTURE.md](ARCHITECTURE.md). Read that first if you're
touching the model or the autograd engine.

## Golden rule: every differentiable op must pass the gradient check

The whole project rests on backprop being correct. `computeMaxGradError()`
compares analytic gradients (from `backward()`) against finite differences and
the tests fail if the relative error exceeds `1e-4`. **Any new op in `Tensor.kt`
must be validated this way** — it's how we catch a wrong derivative.

### Adding a new op

Follow the pattern every op in `Tensor.kt` already uses:

```kotlin
fun myOp(/* ... */): Tensor {
    val a = this
    val out = Tensor(rows, cols)
    // 1) forward: fill out.data from a.data
    // ...
    return out.build(listOf(a)) {
        // 2) backward: read out.grad, ACCUMULATE (+=) into each parent's grad.
        //    d(parent) += d(out) * local-derivative
        for (i in a.data.indices) a.grad[i] += out.grad[i] * /* ∂out_i/∂a_i */ 1.0
    }
}
```

Rules that keep autograd correct:

- **Accumulate, never assign**: use `+=` into `parent.grad`. A tensor can feed
  several ops; its gradient is the sum. `backward()` zeros grads once at the top.
- **List every input** you read in `build(listOf(...))`, so the topo sort visits
  them. Constants (like the causal mask) are not inputs.
- **Don't mutate inputs** in the forward pass; write to a fresh `out`.

Then prove it, e.g. by extending a spot-check like `computeMaxGradError()` or by
adding a focused finite-difference test. If the op affects the model, add or
update a test in `KortexTest.kt`.

## Tests we keep green

`src/test/kotlin/KortexTest.kt` guards the invariants: backprop correctness
(learned and RoPE), tokenizer losslessness, BPE compression, attention causality
and row-normalization, and that training reduces loss. If you change behavior,
update the relevant test in the same PR — don't just loosen an assertion.

Keep tests fast: use tiny configs (small `nEmbed`, `nLayer`, few steps). The full
suite should stay in the seconds range so CI and local runs stay painless.

## Style

- Match the surrounding code: 4-space indent, no wildcard imports, descriptive
  names over abbreviations in public APIs.
- Favor a comment that explains *why* (the math or the intent), not *what* the
  line obviously does. This is a codebase people read to learn.
- No new runtime dependencies. Pure Kotlin stdlib only. (Test-only deps are fine.)
- Keep the "one sequence at a time, scalar `Double`" model unless a change is
  specifically about performance and stays readable.

## Submitting a change

1. Branch off `main`.
2. Make the change; keep commits focused with clear messages.
3. `./gradlew test` passes locally, and `gradcheck` still reports PASS for both
   positional variants if you touched math.
4. Update `README.md` / `ARCHITECTURE.md` if you changed behavior, run modes, or
   structure.
5. Open a PR describing the idea and, for anything numeric, how you verified it.
   CI (`.github/workflows/ci.yml`) runs the test suite on every PR.

## Good first contributions

Ideas that fit the spirit of the project:

- Weight tying between `tokEmb` and the output `head`.
- Dropout (train-time only) with a correct backward.
- A KV-cache to make generation `O(T)` instead of re-encoding each step.
- Top-k / nucleus (top-p) sampling in `Train.kt`.
- A learning-rate warmup + cosine decay schedule for `Adam`.
- Loading a corpus from a file instead of the built-in `corpus()`.

Each should come with a test or a gradient check where applicable.
