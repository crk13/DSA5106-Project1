#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash code/autodl/run_full_eval.sh /root/project/checkpoints/tiny_p5.pth
#
# Optional env vars:
#   AUTODL_PROJECT_ROOT=/root/project

PROJECT_ROOT="${AUTODL_PROJECT_ROOT:-/root/project}"
CODE_DIR="${PROJECT_ROOT}/code"
CHECKPOINT="${1:-${PROJECT_ROOT}/checkpoints/tiny_p5.pth}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
WORK_BASE="${PROJECT_ROOT}/work_dirs/eval_full_${RUN_TAG}"

mkdir -p "${WORK_BASE}"
cd "${CODE_DIR}"

echo "[1/3] Eval odinw..."
python mmdet_test.py \
  "${CODE_DIR}/configs/val/custom_autodl/grounding_dino_swin-t_only_p5_eval_odinw_custom.py" \
  "${CHECKPOINT}" \
  --work-dir "${WORK_BASE}/odinw" \
  --out "${WORK_BASE}/odinw/preds.pkl"

echo "[2/3] Eval brain_tumor..."
python mmdet_test.py \
  "${CODE_DIR}/configs/val/custom_autodl/grounding_dino_swin-t_only_p5_eval_brain_tumor_custom.py" \
  "${CHECKPOINT}" \
  --work-dir "${WORK_BASE}/brain_tumor" \
  --out "${WORK_BASE}/brain_tumor/preds.pkl"

echo "[3/3] Eval tbx11k..."
python mmdet_test.py \
  "${CODE_DIR}/configs/val/custom_autodl/grounding_dino_swin-t_only_p5_eval_tbx11k_custom.py" \
  "${CHECKPOINT}" \
  --work-dir "${WORK_BASE}/tbx11k" \
  --out "${WORK_BASE}/tbx11k/preds.pkl"

python "${CODE_DIR}/autodl/collect_eval_results.py" \
  --work-dirs "${WORK_BASE}/odinw" "${WORK_BASE}/brain_tumor" "${WORK_BASE}/tbx11k" \
  --output-json "${WORK_BASE}/summary_full.json" \
  --output-csv "${WORK_BASE}/summary_full.csv"

echo "Full eval done."
echo "Work dir: ${WORK_BASE}"
echo "Summary:  ${WORK_BASE}/summary_full.csv"

