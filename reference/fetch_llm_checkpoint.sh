#!/usr/bin/env bash
# Fetch qwen3.8-flash-next's UD-Q4_K_XL (4 shards, 111.32 GB) and the MTP head
# (2.79 GB) into models/, which is gitignored — hence this script living here.
#
# Resumable: every file is `curl -C -`'d until its size matches the one the
# Hugging Face API reports, so rerunning after an interruption continues, and
# rerunning after a complete download is a no-op. 18 minutes at ~100 MiB/s.
#
#	reference/fetch_llm_checkpoint.sh
#	go run ./cmd/gguf models/Qwen3.8-Flash-Next-GGUF/…-00001-of-00004.gguf
set -u
B=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main
D=$(cd "$(dirname "$0")/.." && pwd)/models/Qwen3.8-Flash-Next-GGUF
mkdir -p "$D"

want() {
  local rel=$1
  local exp=$2
  local out
  out=$D/$(basename "$rel")
  local try have rc
  for try in $(seq 1 20); do
    have=0
    [ -f "$out" ] && have=$(stat -c%s "$out")
    if [ "$have" = "$exp" ]; then echo "$(date -Is) ok   $(basename "$out") $exp"; return 0; fi
    echo "$(date -Is) get  $(basename "$out") have=$have exp=$exp try=$try"
    curl -sSL -C - --fail --retry 5 --retry-delay 5 -o "$out" "$B/$rel"
    rc=$?
    echo "$(date -Is) curl rc=$rc size=$(stat -c%s "$out" 2>/dev/null || echo 0)"
  done
  echo "$(date -Is) FAIL $(basename "$out")"; return 1
}

N=Qwen3.8-Flash-Next-UD-Q4_K_XL
want "UD-Q4_K_XL/$N-00001-of-00004.gguf" 10946624 || exit 1
want "UD-Q4_K_XL/$N-00002-of-00004.gguf" 49859583136 || exit 1
want "UD-Q4_K_XL/$N-00003-of-00004.gguf" 49376141504 || exit 1
want "UD-Q4_K_XL/$N-00004-of-00004.gguf" 12087983520 || exit 1
want "MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf" 2786204800 || exit 1
# Unsloth's published importance matrix (LLM.md L8c-2): 1852 F32 tensors,
# `<weight>.in_sum2` and `.counts`, 45 chunks of their own calibration set.
# It is what makes "does calibration move the widths" a free question rather
# than a 360 GB one. Note the source name's `_file` suffix.
want "imatrix_unsloth.gguf_file" 580038720 || exit 1
[ -f "$D/imatrix_unsloth.gguf" ] || mv "$D/imatrix_unsloth.gguf_file" "$D/imatrix_unsloth.gguf"
# The perplexity corpus, so the accuracy reference is reproducible: L1's
# PPL = 4.0340 is this file at n_ctx 2048 (research/l1-baseline.md).
W=$(dirname "$D")/wikitext-2-raw
if [ ! -f "$W/wiki.test.raw" ]; then
  echo "$(date -Is) get  wikitext-2-raw"
  curl -sSL --fail -o "$D/../wikitext-2-raw-v1.zip" \
    https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip &&
    unzip -o -q "$D/../wikitext-2-raw-v1.zip" -d "$(dirname "$D")" &&
    rm -f "$D/../wikitext-2-raw-v1.zip"
fi
echo "$(date -Is) ok   wikitext-2-raw/wiki.test.raw $(stat -c%s "$W/wiki.test.raw" 2>/dev/null || echo missing)"

# The vision tower (LLM-VISION.md V0): llama.cpp's mmproj, which is what
# serves, and the HF checkpoint's first shard, which holds all 333
# `model.visual.*` tensors and is the fp32 oracle's weights
# (reference/dump_llm_vision.py) along with the processor's configs.
want "mmproj-BF16.gguf" 907542944 || exit 1
V=$(dirname "$D")/Qwen3.8-Flash-Next-HF-vision
mkdir -p "$V"
for f in config.json preprocessor_config.json video_preprocessor_config.json \
  tokenizer.json tokenizer_config.json chat_template.jinja; do
  [ -f "$V/$f" ] || curl -sSL --fail -o "$V/$f" "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/$f"
done
H=model-00001-of-00131.safetensors
if [ "$(stat -c%s "$V/$H" 2>/dev/null)" != 1040155944 ]; then
  curl -sSL -C - --fail -o "$V/$H" "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/$H"
fi
echo "$(date -Is) ok   $(basename "$V")/$H $(stat -c%s "$V/$H" 2>/dev/null || echo missing)"

echo "$(date -Is) ALL DONE"
