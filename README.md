# minllm — a minimalistic LLM in Kotlin (from scratch)

A tiny **GPT** (decoder-only transformer) written in pure Kotlin — no ML
libraries, no linear-algebra dependencies. Every algorithm an LLM relies on is
here in readable form: tokenization, embeddings, self-attention,
backpropagation, an optimizer, and autoregressive sampling.

It is small enough to read in one sitting and it **trains on a CPU in a couple of
minutes**, yet it is architecturally the same as GPT-2 — just with tiny numbers.

```
tokens ─► token embedding ┐
                          ├─(+)─► [ Transformer Block ] × N ─► LayerNorm ─► logits ─► softmax ─► next token
positions ─► pos embedding┘
```

---

## The files (read them in this order)

| File | What it teaches |
|------|-----------------|
| `src/main/kotlin/Tensor.kt` | **Autograd** — the learning machinery. A `Tensor` holds numbers and, after `backward()`, the gradient of the loss w.r.t. each number. Each op (matmul, softmax, layernorm, cross-entropy…) knows how to send gradients to its inputs. This is backpropagation. |
| `src/main/kotlin/Model.kt` | **The transformer** — token/position embeddings, causal multi-head self-attention, the feed-forward MLP, residual connections, LayerNorm, and how they stack into a GPT. |
| `src/main/kotlin/Tokenizer.kt` | **Tokenization** — a char tokenizer and a byte-level **BPE** tokenizer (the algorithm GPT-2/3 use) behind one interface. |
| `src/main/kotlin/Train.kt` | **Working with the model** — the Adam optimizer, a numerical **gradient check**, the training loop, and text generation (temperature sampling). |
| `src/main/kotlin/Viz.kt` | **Attention visualizer** — renders each head's attention matrix as ASCII + an HTML heatmap, so you can *see* which tokens attend to which. |
| `src/test/kotlin/MinllmTest.kt` | **Tests** — `./gradlew test` checks backprop, tokenizer round-trips, attention causality, and that training lowers the loss. |

---

## Build & run

Uses **Gradle** (a wrapper is included, so you don't even need Gradle installed —
just a JDK 17+; developed on JDK 21). The first run downloads Kotlin.

```bash
# 1) Prove the math is right (analytic gradients vs. finite differences):
./gradlew run --args="gradcheck"
#   -> Gradient check: max relative error = ~3e-06  ... PASS

# 2) Train on the built-in corpus, then generate:
./gradlew run --args="train char"     # character tokenizer (default)
./gradlew run --args="train bpe"      # byte-level BPE tokenizer

# 3) Visualize attention (trains a small model, then draws heatmaps):
./gradlew run --args="attn char"      # writes attention.html + prints ASCII grids

# 4) Run the automated test suite:
./gradlew test
```

### Tests (`src/test/kotlin/MinllmTest.kt`)

`./gradlew test` (JUnit 5) asserts the properties everything depends on:

| Test | Guarantees |
|------|-----------|
| `backpropMatchesNumericalGradients` | analytic gradients ≈ finite differences (backprop is correct) |
| `charTokenizerIsLossless` / `bpeIsLossless` | `decode(encode(x)) == x`, including unseen UTF-8 |
| `bpeCompressesSequence` | BPE produces fewer tokens than characters |
| `attentionIsCausalAndNormalized` | no attention to future tokens; each row sums to 1 |
| `trainingReducesLoss` | a short training run lowers the loss |

Prefer plain Kotlin? It still compiles directly:
`kotlinc src/main/kotlin/*.kt -include-runtime -d minllm.jar && java -jar minllm.jar train char`

### What training looks like

```
tokenizer=char
chars=159  tokens=159  vocab=28  params=~110k
step  250  loss 0.56
step  500  loss 0.12
...
--- samples (temperature 0.8) ---
[to be]     -> to be that is the question. all that glitters is not gold. the qu
[knowledge] -> knowledge is power and power is knowledge is power and power is knowl
```

The corpus is a handful of famous sentences, so the model **memorizes** their
structure — perfect for *seeing* learning happen without waiting hours. Loss
starts near `ln(vocab) ≈ 3.3` (random guessing) and falls toward `0`. With
`bpe`, watch the `tokens=` count drop below `chars=` — BPE packs common
sequences into single tokens, so the model sees more text per step.

### The attention visualizer

`attn` runs one forward pass with a capture hook and draws every head's
attention matrix. Row *i* = the token doing the looking (query); column *j* =
the token being looked at (key). Every grid is **lower-triangular** — the causal
mask forbids attending to the future — which you can literally see:

```
Layer 1 · Head 1
        t h e · q u i c k ...
     t |@
     h |  %
     e |. #
     q |=       =
     k |              *
```

`attention.html` shows the same data as colored heatmaps (hover a cell for the
exact weight). Different heads learn different jobs — some attend to the
previous token, some to word starts, some far back.

---

## The core algorithms, mapped to code

**1. Tokenization** (`CharTokenizer`, `BpeTokenizer`) — text ⇄ integer ids.
Char = one id per character. **BPE** starts from raw bytes and repeatedly merges
the most frequent adjacent pair into a new token, so `"the"`/`" is"` become
single ids. Working on bytes means *any* text encodes (no "unknown token").

**2. Embeddings** (`GPT.tokEmb`, `posEmb`) — each token id and each position gets
a learned vector. `x = tokEmb[token] + posEmb[position]`. Position matters
because attention itself is order-agnostic.

**3. Self-attention** (`Attention.forward`) — the only place tokens exchange
information. For each token we build a **query** ("what am I looking for?"), a
**key** ("what do I offer?"), and a **value** ("what I'll pass on"):

```
scores = softmax( (Q · Kᵀ) / √headDim  + causalMask )      # who attends to whom
out    = scores · V                                        # weighted blend of values
```

The **causal mask** sets attention to future tokens to −∞, so position *i* can
only see positions `0..i`. That single constraint is what makes it a *language*
model (predict-the-next-token) rather than a lookup.
**Multi-head** = do this several times in parallel on slices of the vectors, then
concatenate — each head can specialize (one tracks syntax, another long-range
references, …).

**4. Transformer block** (`Block.forward`) — `x = x + Attention(LN(x))` then
`x = x + MLP(LN(x))`. The `+` is a **residual connection**: each block learns a
*correction*, which keeps gradients flowing through deep stacks. **LayerNorm**
stabilizes the scale of activations. The **MLP** (`d→4d→GELU→d`) is where
per-token "thinking" happens.

**5. The loss** (`Tensor.crossEntropy`) — turn final logits into probabilities
with softmax, then penalize the negative log-probability of the correct next
token. One scalar the whole network minimizes.

**6. Backpropagation** (`Tensor.backward`) — topologically order the graph, seed
`dLoss/dLoss = 1`, and run each op's local gradient rule in reverse. Verified by
`gradcheck`.

**7. Optimization** (`Adam`) — nudge every parameter opposite its gradient, with
per-parameter adaptive step sizes and momentum.

**8. Generation** (`generate` + `sample`) — feed the model its own output one
token at a time. **Temperature** controls randomness: `→0` greedy/repetitive,
`→1+` more diverse.

---

## Experiments to try (this is how you learn)

- **Scale it up.** In `train()`, bump `nLayer`, `nEmbed`, `nHead`, `steps`. Watch
  loss and sample quality change. (Bigger = slower on CPU.)
- **Temperature sweep.** Call `generate(..., temperature = 0.2)` vs `1.2`. See
  greedy repetition vs. creative nonsense.
- **Your own corpus.** Replace the text in `corpus()` (`Train.kt`) with any
  paragraph. The vocab and everything else adapt automatically.
- **Char vs. BPE.** Run `train char` then `train bpe` and compare the `tokens=`
  count and sample quality.
- **Break attention.** Comment out `causalMask` and retrain — the model can now
  "cheat" by seeing the future, and the loss collapses toward 0 but it can't
  generate. This shows *why* the mask exists.
- **Single vs. multi-head.** Set `nHead = 1`. Compare.
- **Watch a head learn.** Run `attn` and open `attention.html`. Retrain for more
  steps and watch the heads sharpen from near-uniform into clear patterns.

---

## How this maps to a real LLM

Same architecture, different scale. A production model uses BPE with ~100k
tokens, billions of parameters and thousands of GPUs, trains on trillions of
tokens, and adds RoPE positions, KV-caching, and post-training (instruction
tuning + RLHF). The *algorithms* on this page — tokenization, attention,
backprop, cross-entropy, Adam, autoregressive sampling — are exactly the ones
running inside them.

*Educational implementation. Prioritizes clarity over speed: plain `Double`
loops, one sequence at a time, no vectorization/GPU.*
