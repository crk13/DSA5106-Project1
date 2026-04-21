#!/usr/bin/env python3
"""Collect key eval metrics from MMEngine log files into JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


COPYPaste_RE = re.compile(
    r"bbox_mAP_copypaste:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
)


def parse_log_text_metrics(log_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = COPYPaste_RE.findall(text)
    if matches:
        last = matches[-1]
        metrics["bbox_mAP"] = float(last[0])
        metrics["bbox_mAP_50"] = float(last[1])
        metrics["bbox_mAP_75"] = float(last[2])
        metrics["bbox_mAP_s"] = float(last[3])
        metrics["bbox_mAP_m"] = float(last[4])
        metrics["bbox_mAP_l"] = float(last[5])
    return metrics


def parse_log_json_metrics(log_json_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    target_suffixes = [
        "bbox_mAP",
        "bbox_mAP_50",
        "bbox_mAP_75",
        "bbox_mAP_s",
        "bbox_mAP_m",
        "bbox_mAP_l",
    ]
    for line in log_json_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        for k, v in obj.items():
            for suffix in target_suffixes:
                if k == suffix or k.endswith("/" + suffix):
                    try:
                        metrics[suffix] = float(v)
                    except Exception:
                        pass
    return metrics


def find_latest_file(paths: List[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def collect_one(work_dir: Path) -> Dict[str, object]:
    logs = list(work_dir.rglob("*.log"))
    log_jsons = list(work_dir.rglob("*.log.json"))
    latest_log = find_latest_file(logs)
    latest_log_json = find_latest_file(log_jsons)

    merged: Dict[str, float] = {}
    if latest_log is not None:
        merged.update(parse_log_text_metrics(latest_log))
    if latest_log_json is not None:
        merged.update(parse_log_json_metrics(latest_log_json))

    return {
        "work_dir": str(work_dir),
        "latest_log": str(latest_log) if latest_log else None,
        "latest_log_json": str(latest_log_json) if latest_log_json else None,
        "metrics": merged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect eval metrics from work_dirs.")
    parser.add_argument(
        "--work-dirs",
        nargs="+",
        required=True,
        help="One or more work_dir paths.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    rows = []
    for wd in args.work_dirs:
        result = collect_one(Path(wd))
        metrics = result["metrics"]
        row = {
            "work_dir": result["work_dir"],
            "bbox_mAP": metrics.get("bbox_mAP"),
            "bbox_mAP_50": metrics.get("bbox_mAP_50"),
            "bbox_mAP_75": metrics.get("bbox_mAP_75"),
            "bbox_mAP_s": metrics.get("bbox_mAP_s"),
            "bbox_mAP_m": metrics.get("bbox_mAP_m"),
            "bbox_mAP_l": metrics.get("bbox_mAP_l"),
            "latest_log": result["latest_log"],
            "latest_log_json": result["latest_log_json"],
        }
        rows.append(row)

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fields = [
        "work_dir",
        "bbox_mAP",
        "bbox_mAP_50",
        "bbox_mAP_75",
        "bbox_mAP_s",
        "bbox_mAP_m",
        "bbox_mAP_l",
        "latest_log",
        "latest_log_json",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote: {out_json}")
    print(f"wrote: {out_csv}")


if __name__ == "__main__":
    main()

