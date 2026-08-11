# Kortex — a minimalistic LLM in Kotlin (from scratch)

[![CI](https://github.com/brayskiy/kortex/actions/workflows/ci.yml/badge.svg)](https://github.com/brayskiy/kortex/actions/workflows/ci.yml)

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

> Deeper dives: **[ARCHITECTURE.md](ARCHITECTURE.md)** (how it fits together,
> data/shape flow, the autograd design), **[CONTRIBUTING.md](CONTRIBUTING.md)**
> (build/test, and how to add a differentiable op that passes the gradient check),
> and **[BRANCHING.md](BRANCHING.md)** (the `work → develop → main` PR flow).
> Release history is in **[CHANGELOG.md](CHANGELOG.md)** (auto-generated from releases).

---

## The files (read them in this order)

| File | What it teaches |
|------|-----------------|
| `src/main/kotlin/Tensor.kt` | **Autograd** — the learning machinery. A `Tensor` holds numbers and, after `backward()`, the gradient of the loss w.r.t. each number. Each op (matmul, softmax, layernorm, cross-entropy, RoPE…) knows how to send gradients to its inputs. This is backpropagation. |
| `src/main/kotlin/Model.kt` | **The transformer** — token/position embeddings (learned **or** RoPE), causal multi-head self-attention, the feed-forward MLP, residual connections, LayerNorm, optional **weight tying** and **dropout**, and how they stack into a GPT. |
| `src/main/kotlin/Tokenizer.kt` | **Tokenization** — a char tokenizer and a byte-level **BPE** tokenizer (the algorithm GPT-2/3 use) behind one interface. |
| `src/main/kotlin/Train.kt` | **Working with the model** — the Adam optimizer, a numerical **gradient check**, the training loop, temperature sampling, and the **CLI** (`train`/`generate`/`chat`). |
| `src/main/kotlin/Checkpoint.kt` | **Save / load** — serialize a trained model (config + tokenizer + weights) to one file and reload it. |
| `src/main/kotlin/Sampling.kt` | **Sampling** — turn logits into a token with temperature, **top-k**, and **top-p** (nucleus). |
| `src/main/kotlin/Inference.kt` | **KV-cache** — fast decoding that caches past keys/values instead of recomputing the whole context each step. |
| `src/main/kotlin/Viz.kt` | **Attention visualizer** — renders each head's attention matrix as ASCII + an HTML heatmap, so you can *see* which tokens attend to which. |
| `src/test/kotlin/KortexTest.kt` | **Tests** — `./gradlew test` checks backprop, tokenizer round-trips, attention causality, and that training lowers the loss. |

---

## Build & run

Uses **Gradle** (a wrapper is included, so you don't even need Gradle installed —
just a JDK 17+; developed on JDK 21). The first run downloads Kotlin.

```bash
# 1) Prove the math is right (analytic gradients vs. finite differences):
./gradlew run --args="gradcheck"
#   -> Gradient check: max relative error = ~3e-06  ... PASS

# 2) Train on the built-in corpus, then generate:
./gradlew run --args="train char"          # character tokenizer (default)
./gradlew run --args="train bpe"           # byte-level BPE tokenizer
./gradlew run --args="train char rope"     # rotary positions instead of a learned table

# 3) Visualize attention (trains a small model, then draws heatmaps):
./gradlew run --args="attn char"           # writes attention.html + prints ASCII grids

# 4) Compare positional encodings (learned absolute vs. RoPE):
./gradlew run --args="poscompare char"

# 5) Run the automated test suite:
./gradlew test
```

## Use the app (train → save → chat)

Kortex has a small CLI: **train** a model to a file, then **generate** from it or
**chat** with it interactively. For an interactive session, build the runnable
binary once (it forwards stdin properly, unlike `gradlew run`):

```bash
./gradlew installDist                 # builds ./build/install/kortex/bin/kortex
BIN=./build/install/kortex/bin/kortex

# Train and save a checkpoint (weights + tokenizer + config -> one file):
$BIN train --steps 1500 --out model.bin
$BIN train --tok bpe --rope --out model.bin        # BPE + rotary positions
$BIN train --tie --dropout 0.1 --out model.bin     # weight tying + dropout
$BIN train --data mytext.txt --embed 96 --layers 3 # your own corpus, bigger model

# One-shot continuation from a saved model:
$BIN generate --model model.bin --prompt "knowledge" --tokens 60 --temp 0.8
$BIN generate --model model.bin --prompt "the" --top-k 1            # greedy
$BIN generate --model model.bin --prompt "the" --temp 1.0 --top-p 0.9   # nucleus
$BIN generate --model model.bin --prompt "the" --kv                 # KV-cache path

# Interactive REPL — type a prompt, it continues it:
$BIN chat --model model.bin
#   > to be
#   to be that is the question. all that glitters is not gold...
#   Commands inside chat:  :temp 0.4   :top-k 5   :top-p 0.9   :tokens 80   :quit
```

**Regularization / efficiency** (train-time): `--tie` shares the token-embedding
matrix as the output projection (`logits = x · tokEmbᵀ`), removing the separate
head — fewer parameters (`vocab × embed`) and often better quality. `--dropout F`
randomly zeros activations during training (disabled at generation) to reduce
overfitting.

**Sampling knobs** (all optional): `--temp` scales randomness (→0 greedy, >1 wild);
`--top-k N` samples only from the N most likely tokens; `--top-p F` (nucleus) keeps
the most likely tokens summing to probability F. top-k and top-p compose.

**KV-cache & speed.** `--kv` decodes with a key/value cache instead of recomputing
the whole context each step — same output, far less work. See it yourself:

```bash
$BIN bench --block 160 --embed 128 --layers 4
#   full recompute :  14847.6 ms  (11 tok/s)
#   KV-cache       :    112.3 ms  (1424 tok/s)
#   speedup        : 132.2x
#   outputs identical: true
```

(The KV path is bounded by the model's context window; for longer sliding-window
output, drop `--kv` and use the default generator.)

`$BIN help` lists every command and flag. Prefer no build step? `./gradlew run
--args="train --steps 800 --out model.bin"` works for one-shot commands (use the
installed binary for `chat`, which needs live stdin).

> This is a tiny character/BPE language model trained on a few sentences — think
> *autocomplete that memorized its corpus*, not an assistant. It continues text;
> it doesn't follow instructions. Turn `--temp` down for faithful recall, up for
> more variety.

---

## Demo modes

Run modes take up to three args: `<mode> [char|bpe] [learned|rope]`.

### Tests (`src/test/kotlin/KortexTest.kt`)

`./gradlew test` (JUnit 5) asserts the properties everything depends on:

| Test | Guarantees |
|------|-----------|
| `backpropMatchesNumericalGradients` | analytic gradients ≈ finite differences (backprop is correct) |
| `charTokenizerIsLossless` / `bpeIsLossless` | `decode(encode(x)) == x`, including unseen UTF-8 |
| `bpeCompressesSequence` | BPE produces fewer tokens than characters |
| `attentionIsCausalAndNormalized` | no attention to future tokens; each row sums to 1 |
| `trainingReducesLoss` | a short training run lowers the loss |
| `backpropMatchesNumericalGradients_rope` | the RoPE rotation backprops correctly |
| `ropeDropsThePositionTable` | RoPE removes exactly the learned position table |

Prefer plain Kotlin? It still compiles directly:
`kotlinc src/main/kotlin/*.kt -include-runtime -d kortex.jar && java -jar kortex.jar train char`

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

**2. Embeddings + positions** (`GPT.tokEmb`, `posEmb`) — each token id gets a
learned vector. Position matters (attention itself is order-agnostic), and Kortex
offers two ways to add it:
- **Learned absolute** (default): a lookup table of position vectors added to the
  token vectors — `x = tokEmb[token] + posEmb[position]`.
- **RoPE / rotary** (`useRope`, `Tensor.rope`): no table at all — inside attention
  the query/key vectors are *rotated* by an angle proportional to their position,
  so the score `Q·K` ends up depending only on the **relative** offset between two
  tokens. Fewer parameters, and it extrapolates to longer contexts.

Run `poscompare` to train both under identical settings — on the toy corpus RoPE
matches or beats the learned table with fewer parameters.

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
