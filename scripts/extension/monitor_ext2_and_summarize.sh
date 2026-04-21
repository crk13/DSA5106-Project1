#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="/autodl-fs/data/llmdet_extension/outputs"
RESULT_ROOT="/autodl-fs/data/llmdet_extension/extension2_results"
RUN_LOG_DIR="${OUTPUT_ROOT}/_run_logs"
MONITOR_LOG_DIR="/root/autodl-tmp/llmdet_monitor_logs"
mkdir -p "${RESULT_ROOT}" "${RUN_LOG_DIR}" "${MONITOR_LOG_DIR}"

EXPS=(
  ext2_raw_long_2k
  ext2_cleaned_long_2k
  ext2_raw_short_2k
  ext2_cleaned_short_2k
)

MONITOR_LOG="${MONITOR_LOG_DIR}/monitor_ext2_$(date +%Y%m%d_%H%M%S).log"

log_msg() {
  local msg="$1"
  echo "$msg"
  printf '%s\n' "$msg" >> "${MONITOR_LOG}" 2>/dev/null || true
}

log_msg "[$(date '+%F %T')] monitor started"

log_msg "[$(date '+%F %T')] waiting for 4 experiments to finish"

exp_done() {
  local exp="$1"
  local ckpt="${OUTPUT_ROOT}/${exp}/work_dir/iter_2000.pth"
  local pred="${OUTPUT_ROOT}/${exp}/eval/predictions.pkl"
  [[ -f "${ckpt}" && -f "${pred}" ]]
}

all_done() {
  local exp
  for exp in "${EXPS[@]}"; do
    if ! exp_done "${exp}"; then
      return 1
    fi
  done
  return 0
}

master_alive() {
  pgrep -f "run_extension2_pilot_4vars.sh|run_extension2_pilot_4vars_resume.sh|mmdet_train.py|mmdet_test.py|dist_train.sh|dist_test.sh" >/dev/null 2>&1
}

progress_line() {
  local latest_log
  latest_log=$(ls -t "${RUN_LOG_DIR}"/ext2_4vars_* "${RUN_LOG_DIR}"/ext2_resume_4vars_* "${RUN_LOG_DIR}"/ext2_resume_fast_* 2>/dev/null | head -n 1 || true)
  if [[ -n "${latest_log}" && -f "${latest_log}" ]]; then
    local it
    it=$(grep -E "Iter\((train|test|val)\) \[[[:space:]]*[0-9]+/[0-9]+\]" "${latest_log}" | tail -n 1 || true)
    if [[ -n "${it}" ]]; then
      echo "${it}"
      return
    fi
  fi

  local latest_eval
  latest_eval=$(find "${OUTPUT_ROOT}" -type f -path "*/eval/*/*.log" 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${latest_eval}" && -f "${latest_eval}" ]]; then
    local eit
    eit=$(grep -E "Iter\((test|val)\) \[[[:space:]]*[0-9]+/[0-9]+\]" "${latest_eval}" | tail -n 1 || true)
    if [[ -n "${eit}" ]]; then
      echo "${eit}"
      return
    fi
  fi

  echo "no iter info yet"
}

while true; do
  done_count=0
  for exp in "${EXPS[@]}"; do
    if exp_done "${exp}"; then
      done_count=$((done_count + 1))
    fi
  done

  log_msg "[$(date '+%F %T')] progress: ${done_count}/4 done, $(progress_line)"

  if all_done; then
    log_msg "[$(date '+%F %T')] all experiments completed"
    break
  fi

  if ! master_alive; then
    log_msg "[$(date '+%F %T')] training process ended before all_done; generating partial summary"
    break
  fi

  sleep 120
done

SUMMARY_MD="${RESULT_ROOT}/report_materials_$(date +%Y%m%d_%H%M%S).md"
LATEST_MD="${RESULT_ROOT}/report_materials_latest.md"
export SUMMARY_MD
export LATEST_MD

python - << 'PY'
import csv
import glob
import json
import os
import re
from datetime import datetime

output_root = "/autodl-fs/data/llmdet_extension/outputs"
result_root = "/autodl-fs/data/llmdet_extension/extension2_results"
summary_md = os.environ["SUMMARY_MD"]
latest_md = os.environ["LATEST_MD"]

exps = [
    "ext2_raw_long_2k",
    "ext2_cleaned_long_2k",
    "ext2_raw_short_2k",
    "ext2_cleaned_short_2k",
]

iter_re = re.compile(r"Iter\(train\) \[\s*(\d+)/(\d+)\]")
float_re = re.compile(r"[-+]?\d*\.\d+|\d+")


def latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def read_text(path):
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def parse_last_iter(log_text):
    matches = iter_re.findall(log_text)
    if not matches:
        return ""
    a, b = matches[-1]
    return f"{a}/{b}"


def parse_last_loss(log_text):
    lines = [ln for ln in log_text.splitlines() if " loss:" in ln or "loss:" in ln]
    if not lines:
        return ""
    line = lines[-1]
    m = re.search(r"\bloss:\s*([0-9]*\.?[0-9]+)", line)
    return m.group(1) if m else ""


def parse_eval_metrics(eval_text):
    keys = [
        "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l",
        "AP", "AP50", "AP75", "APr", "APc", "APf",
    ]
    found = {}

    for line in eval_text.splitlines():
        if "copypaste" in line:
            nums = [x for x in float_re.findall(line)]
            if len(nums) >= 6:
                found.setdefault("AP", nums[-6])
                found.setdefault("AP50", nums[-5])
                found.setdefault("AP75", nums[-4])
                found.setdefault("APr", nums[-3])
                found.setdefault("APc", nums[-2])
                found.setdefault("APf", nums[-1])
        for k in keys:
            if k in line:
                m = re.search(rf"{re.escape(k)}\s*[:=]\s*([0-9]*\.?[0-9]+)", line)
                if m:
                    found[k] = m.group(1)

    return found


rows = []
for exp in exps:
    exp_dir = os.path.join(output_root, exp)
    work_dir = os.path.join(exp_dir, "work_dir")
    eval_dir = os.path.join(exp_dir, "eval")

    ckpt = os.path.join(work_dir, "iter_2000.pth")
    pred = os.path.join(eval_dir, "predictions.pkl")

    train_log = latest_file(os.path.join(work_dir, "*", "*.log"))
    eval_log = latest_file(os.path.join(eval_dir, "*.log"))

    train_text = read_text(train_log)
    eval_text = read_text(eval_log)

    last_iter = parse_last_iter(train_text)
    last_loss = parse_last_loss(train_text)
    metrics = parse_eval_metrics(eval_text)

    analysis_pngs = sorted(glob.glob(os.path.join(exp_dir, "analysis_current", "*.png")))

    rows.append({
        "exp": exp,
        "ckpt_done": os.path.isfile(ckpt),
        "eval_done": os.path.isfile(pred),
        "last_iter": last_iter,
        "last_loss": last_loss,
        "train_log": train_log,
        "eval_log": eval_log,
        "metrics": metrics,
        "analysis_pngs": analysis_pngs,
    })

def rel(p):
    return p if p else ""

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Extension II 自动汇总（4组）\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("## 1) 实验完成状态\n\n")
    f.write("| 实验 | ckpt(iter_2000) | eval(predictions.pkl) | 最后迭代 | 最后loss |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['exp']} | {'Y' if r['ckpt_done'] else 'N'} | {'Y' if r['eval_done'] else 'N'} | {r['last_iter'] or '-'} | {r['last_loss'] or '-'} |\n"
        )

    f.write("\n## 2) 可写入 Report 的关键指标（从评估日志自动提取）\n\n")
    all_keys = ["AP", "AP50", "AP75", "APr", "APc", "APf", "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l"]
    f.write("| 实验 | " + " | ".join(all_keys) + " |\n")
    f.write("|---|" + "|".join(["---:"] * len(all_keys)) + "|\n")
    for r in rows:
        vals = [r["metrics"].get(k, "-") for k in all_keys]
        f.write(f"| {r['exp']} | " + " | ".join(vals) + " |\n")

    f.write("\n## 3) 推荐放入 Report 的图\n\n")
    f.write("优先级建议:\n")
    f.write("- 必放: 4组 AP/APr/APc/APf 对比柱状图（从上表整理）\n")
    f.write("- 必放: 4组 loss 曲线对比图（同一坐标轴）\n")
    f.write("- 建议: 4组 lr 曲线与 grad_norm 曲线\n")
    f.write("- 建议: 代表性可视化 case（正确/错误各2-3例）\n\n")

    for r in rows:
        f.write(f"### {r['exp']}\n")
        if r["analysis_pngs"]:
            for p in r["analysis_pngs"]:
                f.write(f"- {rel(p)}\n")
        else:
            f.write("- 暂无 analysis_current/*.png（可后处理生成）\n")
        f.write("\n")

    f.write("## 4) 原始日志路径\n\n")
    for r in rows:
        f.write(f"- {r['exp']}\n")
        f.write(f"  - train_log: {rel(r['train_log']) or '-'}\n")
        f.write(f"  - eval_log: {rel(r['eval_log']) or '-'}\n")

with open(latest_md, "w", encoding="utf-8") as f:
    with open(summary_md, "r", encoding="utf-8") as src:
        f.write(src.read())

print(summary_md)
print(latest_md)
PY

log_msg "[$(date '+%F %T')] summary generated: ${SUMMARY_MD}"
log_msg "[$(date '+%F %T')] latest summary: ${LATEST_MD}"
log_msg "[$(date '+%F %T')] monitor finished"
