#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/extension/run_extension_exp.sh <config> <gpus> <resume_ckpt_or_none>"
  echo "Example: bash scripts/extension/run_extension_exp.sh configs/extension_caps/exp_template.py 1 none"
  exit 1
fi

CFG="$1"
GPUS="$2"
RESUME="$3"

if [[ "$RESUME" == "none" ]]; then
  bash dist_train.sh "$CFG" "$GPUS" --amp
else
  bash dist_train.sh "$CFG" "$GPUS" --amp --resume "$RESUME"
fi
