from pathlib import Path
import os
import re
import json
import pickle
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze LLMDet training artifacts and visualize logs."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="/root/LLMDet",
        help="Root directory of the LLMDet project."
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/root/LLMDet/work_dirs/grounding_dino_t_for_alignment",
        help="Work directory to analyze."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save analysis outputs. Default: <work_dir>/analysis"
    )
    parser.add_argument(
        "--pred-pkl",
        type=str,
        default=None,
        help="Optional path to predictions.pkl for future offline evaluation."
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")


def inspect_work_dir(work_dir: Path):
    rows = []
    if not work_dir.exists():
        return pd.DataFrame(rows)

    for p in sorted(work_dir.iterdir()):
        row = {
            "name": p.name,
            "path": str(p),
            "type": "dir" if p.is_dir() else "file",
            "size_mb": None if p.is_dir() else round(p.stat().st_size / 1024 / 1024, 4)
        }
        rows.append(row)
    return pd.DataFrame(rows)


def read_last_checkpoint(work_dir: Path):
    last_ckpt_file = work_dir / "last_checkpoint"
    if last_ckpt_file.exists():
        return last_ckpt_file.read_text(encoding="utf-8").strip()
    return None


def parse_iter_from_ckpt_name(name: str):
    m = re.search(r"iter_(\d+)\.pth", name)
    if m:
        return int(m.group(1))
    return None


def list_checkpoints(work_dir: Path):
    ckpts = sorted(work_dir.glob("*.pth"))
    rows = []
    for p in ckpts:
        rows.append({
            "checkpoint": p.name,
            "iter": parse_iter_from_ckpt_name(p.name),
            "path": str(p),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 4)
        })
    df = pd.DataFrame(rows)
    if not df.empty and "iter" in df.columns:
        df = df.sort_values(["iter", "checkpoint"], ascending=[True, True]).reset_index(drop=True)
    return df


def find_log_files(work_dir: Path):
    files = []
    for ext in ["*.log", "*.json", "*.txt"]:
        files.extend(work_dir.rglob(ext))
    files = sorted(set(files))
    rows = []
    for p in files:
        try:
            rel = p.relative_to(work_dir)
        except Exception:
            rel = p
        rows.append({
            "name": p.name,
            "relative_path": str(rel),
            "full_path": str(p),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 4)
        })
    return pd.DataFrame(rows)


def preview_text_file(path: Path, n=20):
    lines = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line.rstrip("\n"))
    except Exception as e:
        lines.append(f"[ERROR] Failed to read file: {e}")
    return "\n".join(lines)


pattern_iter = re.compile(r"Iter\(train\) \[\s*(\d+)/(\d+)\]")
pattern_kv = re.compile(r"([A-Za-z_][A-Za-z0-9_\.]*):\s*([0-9eE\+\-\.]+)")


def parse_mmengine_log_text(log_path: Path):
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Iter(train)" not in line:
                continue

            m = pattern_iter.search(line)
            if not m:
                continue

            row = {
                "iter": int(m.group(1)),
                "max_iter": int(m.group(2)),
            }

            for k, v in pattern_kv.findall(line):
                # keep only meaningful metric-like keys
                if k in {"base_lr", "lr", "time", "data_time", "memory", "loss",
                         "loss_cls", "loss_bbox", "loss_iou", "grad_norm"} \
                   or k.startswith(("d0.", "d1.", "d2.", "d3.", "d4.", "enc_", "dn_")):
                    try:
                        row[k] = float(v)
                    except Exception:
                        pass

            rows.append(row)

    return pd.DataFrame(rows)


def parse_and_merge_training_logs(log_df: pd.DataFrame):
    parsed_list = []
    parsed_sources = []

    if log_df.empty:
        return None, []

    for _, row in log_df.iterrows():
        p = Path(row["full_path"])
        if p.suffix not in {".log", ".txt"}:
            continue
        try:
            df_try = parse_mmengine_log_text(p)
            if len(df_try) > 0:
                df_try["source_log"] = str(p)
                parsed_list.append(df_try)
                parsed_sources.append(str(p))
        except Exception:
            continue

    if not parsed_list:
        return None, []

    merged = pd.concat(parsed_list, ignore_index=True)

    # keep the latest occurrence for duplicated iterations
    merged = merged.sort_values(["iter"]).drop_duplicates(subset=["iter"], keep="last")
    merged = merged.sort_values("iter").reset_index(drop=True)

    return merged, parsed_sources


def plot_training_curves(df: pd.DataFrame, output_dir: Path):
    cols_to_plot = ["loss", "loss_cls", "loss_bbox", "loss_iou", "grad_norm", "lr"]
    available = [c for c in cols_to_plot if c in df.columns]

    saved_paths = []
    for col in available:
        plt.figure(figsize=(8, 4))
        plt.plot(df["iter"], df[col])
        plt.xlabel("Iteration")
        plt.ylabel(col)
        plt.title(f"{col} vs Iteration")
        plt.grid(True, alpha=0.3)
        out_path = output_dir / f"{col}_curve.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        saved_paths.append(out_path)

    return saved_paths


def build_training_summary(df: pd.DataFrame):
    summary = {}
    summary["n_records"] = len(df)
    summary["first_iter"] = int(df["iter"].min())
    summary["last_iter"] = int(df["iter"].max())

    for col in ["loss", "loss_cls", "loss_bbox", "loss_iou", "grad_norm", "lr"]:
        if col in df.columns:
            summary[f"{col}_first"] = float(df[col].iloc[0])
            summary[f"{col}_last"] = float(df[col].iloc[-1])
            summary[f"{col}_min"] = float(df[col].min())
            summary[f"{col}_max"] = float(df[col].max())

    return pd.DataFrame([summary])


def build_checkpoint_metric_table(parsed_df: pd.DataFrame, ckpt_df: pd.DataFrame):
    """
    For each checkpoint iter_N.pth, find the nearest matching iteration row in parsed log.
    """
    rows = []
    if parsed_df is None or parsed_df.empty or ckpt_df.empty:
        return pd.DataFrame(rows)

    available_metrics = ["loss", "loss_cls", "loss_bbox", "loss_iou", "grad_norm", "lr"]

    for _, ckpt_row in ckpt_df.iterrows():
        ckpt_iter = ckpt_row["iter"]
        if pd.isna(ckpt_iter):
            continue

        matched = parsed_df.loc[parsed_df["iter"] == ckpt_iter]
        if matched.empty:
            # fallback: use the nearest logged iteration not greater than checkpoint iter
            matched = parsed_df.loc[parsed_df["iter"] <= ckpt_iter].tail(1)
        if matched.empty:
            continue

        r = matched.iloc[0]
        row = {
            "checkpoint": ckpt_row["checkpoint"],
            "iter": int(ckpt_iter),
            "path": ckpt_row["path"],
            "size_mb": ckpt_row["size_mb"],
            "matched_log_iter": int(r["iter"])
        }
        for m in available_metrics:
            row[m] = float(r[m]) if m in matched.columns and pd.notna(r[m]) else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("iter").reset_index(drop=True)
    return df


def plot_checkpoint_comparison(ckpt_metric_df: pd.DataFrame, output_dir: Path):
    saved = []
    if ckpt_metric_df.empty:
        return saved

    cols = ["loss", "loss_cls", "loss_bbox", "loss_iou", "grad_norm", "size_mb"]
    available = [c for c in cols if c in ckpt_metric_df.columns and ckpt_metric_df[c].notna().any()]

    x = ckpt_metric_df["iter"].astype(int).tolist()

    for col in available:
        plt.figure(figsize=(8, 4))
        plt.plot(x, ckpt_metric_df[col], marker="o")
        for _, row in ckpt_metric_df.iterrows():
            if pd.notna(row[col]):
                plt.annotate(
                    row["checkpoint"],
                    (row["iter"], row[col]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8
                )
        plt.xlabel("Checkpoint Iteration")
        plt.ylabel(col)
        plt.title(f"{col} across checkpoints")
        plt.grid(True, alpha=0.3)
        out_path = output_dir / f"checkpoint_compare_{col}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        saved.append(out_path)

    return saved


def inspect_predictions_pkl(pred_pkl: Path):
    if not pred_pkl.exists():
        return {
            "exists": False,
            "message": "predictions.pkl not found"
        }

    info = {
        "exists": True,
        "path": str(pred_pkl)
    }

    try:
        with open(pred_pkl, "rb") as f:
            preds = pickle.load(f)

        info["type"] = str(type(preds))

        try:
            info["length"] = len(preds)
        except Exception:
            info["length"] = None

        try:
            first = preds[0]
            info["first_item_type"] = str(type(first))
            if isinstance(first, dict):
                info["first_item_keys"] = list(first.keys())
            else:
                attrs = [a for a in dir(first) if not a.startswith("_")]
                info["first_item_attrs_head"] = attrs[:30]
        except Exception as e:
            info["first_item_error"] = str(e)

    except Exception as e:
        info["load_error"] = str(e)

    return info


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir) if args.output_dir else work_dir / "analysis"
    ensure_dir(output_dir)

    report_lines = []
    report_lines.append("LLMDet Training Analysis Report")
    report_lines.append("=" * 80)
    report_lines.append(f"PROJECT_ROOT: {project_root}")
    report_lines.append(f"WORK_DIR:     {work_dir}")
    report_lines.append(f"OUTPUT_DIR:   {output_dir}")
    report_lines.append("")

    # 1. Inspect work_dir
    work_df = inspect_work_dir(work_dir)
    work_csv = output_dir / "work_dir_inventory.csv"
    work_df.to_csv(work_csv, index=False)

    report_lines.append("[1] Work directory inventory")
    if work_df.empty:
        report_lines.append("Work directory does not exist or is empty.")
    else:
        report_lines.append(work_df.to_string(index=False))
    report_lines.append("")

    # 2. Read last checkpoint
    last_ckpt = read_last_checkpoint(work_dir)
    report_lines.append("[2] last_checkpoint")
    report_lines.append(str(last_ckpt) if last_ckpt else "No last_checkpoint file found.")
    report_lines.append("")

    # 3. List checkpoints
    ckpt_df = list_checkpoints(work_dir)
    ckpt_csv = output_dir / "checkpoints.csv"
    ckpt_df.to_csv(ckpt_csv, index=False)

    report_lines.append("[3] Checkpoints")
    if ckpt_df.empty:
        report_lines.append("No checkpoint files found.")
    else:
        report_lines.append(ckpt_df.to_string(index=False))
    report_lines.append("")

    # 4. Find log files
    log_df = find_log_files(work_dir)
    log_csv = output_dir / "log_files.csv"
    log_df.to_csv(log_csv, index=False)

    report_lines.append("[4] Log files")
    if log_df.empty:
        report_lines.append("No log files found.")
    else:
        report_lines.append(log_df.to_string(index=False))
    report_lines.append("")

    # 5. Preview some logs
    report_lines.append("[5] Log previews")
    if log_df.empty:
        report_lines.append("No log files available for preview.")
    else:
        for _, row in log_df.head(5).iterrows():
            p = Path(row["full_path"])
            if p.suffix in {".log", ".txt"}:
                report_lines.append("-" * 80)
                report_lines.append(f"Preview of: {p}")
                report_lines.append(preview_text_file(p, n=12))
    report_lines.append("")

    # 6. Parse training log
    parsed_df, parsed_from = parse_and_merge_training_logs(log_df)
    report_lines.append("[6] Training log parsing")
    if parsed_df is None or parsed_df.empty:
        report_lines.append("No parsable MMEngine training text log found.")
    else:
        parsed_csv = output_dir / "parsed_training_log.csv"
        parsed_df.to_csv(parsed_csv, index=False)

        report_lines.append("Parsed from logs:")
        if isinstance(parsed_from, list):
            for src in parsed_from:
                report_lines.append(f"  - {src}")
        else:
            report_lines.append(str(parsed_from))
        report_lines.append(f"Number of parsed rows: {len(parsed_df)}")
        report_lines.append("Head:")
        report_lines.append(parsed_df.head().to_string(index=False))

        # 7. Plot curves
        saved_plots = plot_training_curves(parsed_df, output_dir)
        report_lines.append("")
        report_lines.append("[7] Saved training plots")
        for p in saved_plots:
            report_lines.append(str(p))

        # 8. Summary
        summary_df = build_training_summary(parsed_df)
        summary_csv = output_dir / "training_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        report_lines.append("")
        report_lines.append("[8] Training summary")
        report_lines.append(summary_df.to_string(index=False))

        # 9. Checkpoint-level comparison
        ckpt_metric_df = build_checkpoint_metric_table(parsed_df, ckpt_df)
        ckpt_metric_csv = output_dir / "checkpoint_metric_comparison.csv"
        ckpt_metric_df.to_csv(ckpt_metric_csv, index=False)

        report_lines.append("")
        report_lines.append("[9] Checkpoint metric comparison")
        if ckpt_metric_df.empty:
            report_lines.append("No checkpoint-level comparison could be built.")
        else:
            report_lines.append(ckpt_metric_df.to_string(index=False))

            compare_plots = plot_checkpoint_comparison(ckpt_metric_df, output_dir)
            report_lines.append("")
            report_lines.append("[10] Saved checkpoint comparison plots")
            for p in compare_plots:
                report_lines.append(str(p))

    report_lines.append("")

    # 11. Optional predictions.pkl inspection
    pred_pkl = Path(args.pred_pkl) if args.pred_pkl else (work_dir / "predictions.pkl")
    pred_info = inspect_predictions_pkl(pred_pkl)
    pred_json = output_dir / "predictions_pkl_inspection.json"
    pred_json.write_text(json.dumps(pred_info, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines.append("[11] predictions.pkl inspection")
    report_lines.append(json.dumps(pred_info, indent=2, ensure_ascii=False))
    report_lines.append("")

    # Save final report
    report_path = output_dir / "analysis_report.txt"
    save_text("\n".join(report_lines), report_path)

    print("=" * 80)
    print("Analysis finished.")
    print(f"Report saved to: {report_path}")
    print(f"Inventory CSV:   {work_csv}")
    print(f"Checkpoint CSV:  {ckpt_csv}")
    print(f"Log file CSV:    {log_csv}")

    if parsed_df is not None and not parsed_df.empty:
        print(f"Parsed log CSV:  {output_dir / 'parsed_training_log.csv'}")
        print(f"Summary CSV:     {output_dir / 'training_summary.csv'}")
        print(f"Ckpt metric CSV: {output_dir / 'checkpoint_metric_comparison.csv'}")
        print("Saved plots:")
        for p in sorted(output_dir.glob("*curve.png")):
            print(" -", p)
        for p in sorted(output_dir.glob("checkpoint_compare_*.png")):
            print(" -", p)

    print(f"Prediction info: {pred_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()