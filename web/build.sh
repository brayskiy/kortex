#!/usr/bin/env bash
# Build the self-contained browser chat page: train a small char model, export it
# to JSON, and inline it into chat.template.html -> web/chat.html.
#
# The result runs entirely client-side (JS inference over embedded weights) and
# can be opened directly or published as an Artifact. Open web/chat.html.
#
# Usage:  web/build.sh [steps] [embed] [layers]
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-2500}"; EMBED="${2:-64}"; LAYERS="${3:-2}"
BIN=./build/install/kortex/bin/kortex

echo "==> building the CLI"
./gradlew -q installDist

echo "==> training a char model (memorizing the sample corpus for a lively demo)"
"$BIN" train --sample --steps "$STEPS" --embed "$EMBED" --layers "$LAYERS" \
  --dropout 0.0 --val 0 --eval-every "$((STEPS/5))" --out web/model.bin

echo "==> exporting weights to JSON"
"$BIN" export --model web/model.bin --out web/model.json

echo "==> inlining weights into the page"
python3 - <<'PY'
tpl = open("web/chat.template.html").read()
model = open("web/model.json").read().strip()
open("web/chat.html", "w").write(tpl.replace("__MODEL_JSON__", model))
import os
print("wrote web/chat.html (%.1f MB)" % (os.path.getsize("web/chat.html") / 1e6))
PY

echo "==> done. Open web/chat.html in a browser."
