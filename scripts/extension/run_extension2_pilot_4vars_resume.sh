#!/usr/bin/env bash
set -euo pipefail

RUN=${RUN:-1}
ONLY=${ONLY:-all}   # all | train | eval
GPUS=${GPUS:-1}
MAX_ITERS=${MAX_ITERS:-2000}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TRAIN_CFG_OPTIONS=${TRAIN_CFG_OPTIONS:-}
BASE_PORT=${BASE_PORT:-29600}
VERIFY_CKPT=${VERIFY_CKPT:-1}
EVAL_CFG_OPTIONS=${EVAL_CFG_OPTIONS:-}

ROOT=/root/LLMDet
VAL_CFG=${VAL_CFG:-configs/val/grounding_dino_swin_t_lvis_val.py}

declare -a CFGS=(
  configs/extension_caps/ext2_raw_long_2k.py
  configs/extension_caps/ext2_cleaned_long_2k.py
  configs/extension_caps/ext2_raw_short_2k.py
  configs/extension_caps/ext2_cleaned_short_2k.py
)

run_cmd() {
  local cmd="$1"
  echo "$cmd"
  if [[ "$RUN" == "1" ]]; then
    eval "$cmd"
  fi
}

is_checkpoint_valid() {
  local ckpt_path="$1"
  if [[ ! -f "$ckpt_path" ]]; then
    return 1
  fi
  if [[ "$VERIFY_CKPT" != "1" ]]; then
    return 0
  fi
  python - <<'PY' "$ckpt_path"
import sys
import torch

path = sys.argv[1]
try:
    torch.load(path, map_location='cpu')
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

cd "$ROOT"

for cfg in "${CFGS[@]}"; do
  idx=${idx:-0}
  exp_port=$((BASE_PORT + idx))
  exp_name=$(basename "$cfg" .py)
  work_dir="/autodl-fs/data/llmdet_extension/outputs/${exp_name}/work_dir"
  eval_dir="/autodl-fs/data/llmdet_extension/outputs/${exp_name}/eval"
  ckpt="${work_dir}/iter_${MAX_ITERS}.pth"
  pred="${eval_dir}/predictions.pkl"
  override_opts="train_cfg.max_iters=${MAX_ITERS} train_cfg.val_interval=${MAX_ITERS} default_hooks.checkpoint.interval=${MAX_ITERS}"

  if [[ "$ONLY" == "all" || "$ONLY" == "train" ]]; then
    if is_checkpoint_valid "$ckpt"; then
      echo "[resume] skip train for ${exp_name}, found ${ckpt}"
    else
      if [[ -f "$ckpt" ]]; then
        echo "[resume] invalid checkpoint detected for ${exp_name}, removing ${ckpt}"
        run_cmd "rm -f ${ckpt}"
      fi
      cmd="PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} bash dist_train.sh ${cfg} ${GPUS} --amp --work-dir ${work_dir} --cfg-options ${override_opts}"
      cmd="PORT=${exp_port} ${cmd}"
      if [[ -n "${TRAIN_CFG_OPTIONS}" ]]; then
        cmd="${cmd} ${TRAIN_CFG_OPTIONS}"
      fi
      run_cmd "${cmd}"
    fi
  fi

  if [[ "$ONLY" == "all" || "$ONLY" == "eval" ]]; then
    if [[ -f "$pred" ]]; then
      echo "[resume] skip eval for ${exp_name}, found ${pred}"
    else
      run_cmd "mkdir -p ${eval_dir}"
      eval_cmd="PORT=${exp_port} bash dist_test.sh ${VAL_CFG} ${ckpt} ${GPUS} --work-dir ${eval_dir} --out ${eval_dir}/predictions.pkl"
      if [[ -n "${EVAL_CFG_OPTIONS}" ]]; then
        eval_cmd="${eval_cmd} --cfg-options ${EVAL_CFG_OPTIONS}"
      fi
      run_cmd "${eval_cmd}"
    fi
  fi
  idx=$((idx + 1))
done

echo "Done resume. RUN=${RUN}, ONLY=${ONLY}, GPUS=${GPUS}"
