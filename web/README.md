# Browser demo

`chat.html` runs a trained Kortex model **entirely in the browser** — the weights
are embedded in the page and inference (attention, MLP, sampling) is reimplemented
in ~120 lines of JavaScript. No server, no API. It also draws the model's live
next-character probability distribution as it generates.

## Build it

```bash
web/build.sh            # trains a char model, exports JSON, inlines -> web/chat.html
# or tune: web/build.sh <steps> <embed> <layers>
open web/chat.html      # (or just double-click it)
```

## How it works

- `../src/main/kotlin/…` trains a char-tokenizer GPT and `export` dumps it to JSON
  (`exportJson` in `Train.kt`): the vocab, config, and every weight array.
- `chat.template.html` is the page with a `__MODEL_JSON__` placeholder; the build
  script inlines the exported model so the page is fully self-contained.
- The JS `forward()` mirrors `Model.forward` exactly and `sampleFrom()` mirrors
  `Sampling.kt`. The port is verified: greedy JS output matches the Kotlin model
  character-for-character.

Only the **char** tokenizer is supported for export (BPE would need its merge
logic in JS too). Char-level keeps the page simple and the vocab tiny.

Generated files (`chat.html`, `model.bin`, `model.json`) are git-ignored — run the
build to regenerate them.
