#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/extension/run_extension2_pilot_4vars.sh            # dry-run
#   RUN=1 bash scripts/extension/run_extension2_pilot_4vars.sh      # execute train+eval
#   RUN=1 ONLY=train bash scripts/extension/run_extension2_pilot_4vars.sh
#   RUN=1 ONLY=eval  bash scripts/extension/run_extension2_pilot_4vars.sh

RUN=${RUN:-0}
ONLY=${ONLY:-all}   # all | train | eval
GPUS=${GPUS:-1}
MAX_ITERS=${MAX_ITERS:-2000}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TRAIN_CFG_OPTIONS=${TRAIN_CFG_OPTIONS:-}

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

cd "$ROOT"

for cfg in "${CFGS[@]}"; do
  exp_name=$(basename "$cfg" .py)
  work_dir="/autodl-fs/data/llmdet_extension/outputs/${exp_name}/work_dir"
  eval_dir="/autodl-fs/data/llmdet_extension/outputs/${exp_name}/eval"
  ckpt="${work_dir}/iter_${MAX_ITERS}.pth"
  override_opts="train_cfg.max_iters=${MAX_ITERS} train_cfg.val_interval=${MAX_ITERS} default_hooks.checkpoint.interval=${MAX_ITERS}"

  if [[ "$ONLY" == "all" || "$ONLY" == "train" ]]; then
    cmd="PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} bash dist_train.sh ${cfg} ${GPUS} --amp --work-dir ${work_dir} --cfg-options ${override_opts}"
    if [[ -n "${TRAIN_CFG_OPTIONS}" ]]; then
      cmd="${cmd} ${TRAIN_CFG_OPTIONS}"
    fi
    run_cmd "${cmd}"
  fi

  if [[ "$ONLY" == "all" || "$ONLY" == "eval" ]]; then
    run_cmd "mkdir -p ${eval_dir}"
    run_cmd "bash dist_test.sh ${VAL_CFG} ${ckpt} ${GPUS} --work-dir ${eval_dir} --out ${eval_dir}/predictions.pkl"
  fi
done

echo "Done. RUN=${RUN}, ONLY=${ONLY}, GPUS=${GPUS}"
