#!/usr/bin/env bash
# OmniDocBench's end-to-end evaluation of one prediction folder, for OCR.md
# stage O10, in the benchmark's own Docker image (Python 3.10, TeX Live,
# ImageMagick and Ghostscript for CDM), with the README's config: text edit
# distance, formula edit distance and CDM, table TEDS and edit distance,
# reading order.
#
#   reference/omnidocbench_eval.sh GT.json PRED_DIR OUT_DIR
#
# OUT_DIR gets the evaluator's result/ files; the summary is
# OUT_DIR/*_metric_result.json.
set -euo pipefail
gt=$(realpath "$1") pred=$(realpath "$2") out=$(realpath -m "$3")
mkdir -p "$out"
workers=${WORKERS:-8}
docker run --rm --entrypoint bash \
  -v "$gt":/workspace/gt/gt.json:ro \
  -v "$pred":/workspace/data_md/predictions:ro \
  -v "$out":/workspace/result \
  ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204 \
  -c "cat > configs/o10.yaml << EOF
end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
      cdm_workers: $workers
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: $workers
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./gt/gt.json
    prediction:
      data_path: ./data_md/predictions
    match_method: quick_match
    match_workers: $workers
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
EOF
python pdf_validation.py --config configs/o10.yaml"
