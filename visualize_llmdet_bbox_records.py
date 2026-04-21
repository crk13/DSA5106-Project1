from pathlib import Path
import argparse
import json
import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize record-level and bbox-level statistics for LLMDet predictions.pkl"
    )
    parser.add_argument("--pred-pkl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        return x.tolist()
    except Exception:
        pass
    try:
        return list(x)
    except Exception:
        return []


def extract_pred_instances(item):
    if not isinstance(item, dict):
        return [], [], [], [], []

    pi = item.get("pred_instances", None)
    if pi is None or not isinstance(pi, dict):
        return [], [], [], [], []

    bboxes = to_list(pi.get("bboxes", []))
    scores = to_list(pi.get("scores", []))
    labels = to_list(pi.get("labels", []))
    label_names = to_list(pi.get("label_names", []))
    bbox_index = to_list(pi.get("bbox_index", []))
    return bboxes, scores, labels, label_names, bbox_index


def plot_hist(values, xlabel, title, out_path, bins=50):
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bar(labels, values, xlabel, title, out_path):
    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    pred_path = Path(args.pred_pkl)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    print("Loading predictions.pkl ...")
    with open(pred_path, "rb") as f:
        preds = pickle.load(f)
    print("Loaded.")
    print("n_images =", len(preds))

    image_rows = []

    all_record_scores = []
    all_unique_bbox_top1_scores = []

    label_counter_records = Counter()
    label_counter_unique_top1 = Counter()

    for i, item in enumerate(preds):
        if i % 500 == 0:
            print(f"Processing {i}/{len(preds)}")

        bboxes, scores, labels, label_names, bbox_index = extract_pred_instances(item)

        n_records = len(scores)
        n_unique_bbox = len(set(bbox_index)) if len(bbox_index) > 0 else 0
        ratio = (n_records / n_unique_bbox) if n_unique_bbox > 0 else None

        image_rows.append({
            "image_index": i,
            "img_id": item.get("img_id", None) if isinstance(item, dict) else None,
            "img_path": item.get("img_path", None) if isinstance(item, dict) else None,
            "text_len": len(item.get("text", [])) if isinstance(item, dict) else None,
            "record_count": n_records,
            "unique_bbox_count": n_unique_bbox,
            "records_per_unique_bbox": ratio,
            "score_mean_record": (sum(scores) / len(scores)) if len(scores) > 0 else None,
            "score_max_record": max(scores) if len(scores) > 0 else None,
            "score_min_record": min(scores) if len(scores) > 0 else None,
        })

        all_record_scores.extend(scores)

        # record-level label stats
        if len(label_names) == len(scores):
            label_counter_records.update(label_names)
        elif len(labels) == len(scores):
            label_counter_records.update([str(x) for x in labels])

        # unique-bbox-level top1 score / label
        best_per_bbox = {}
        for j in range(len(scores)):
            bi = bbox_index[j]
            sc = float(scores[j])
            lab_name = label_names[j] if j < len(label_names) else str(labels[j])

            if bi not in best_per_bbox or sc > best_per_bbox[bi]["score"]:
                best_per_bbox[bi] = {
                    "score": sc,
                    "label_name": lab_name
                }

        for _, info in best_per_bbox.items():
            all_unique_bbox_top1_scores.append(info["score"])
            label_counter_unique_top1.update([info["label_name"]])

    image_df = pd.DataFrame(image_rows)
    image_df.to_csv(out_dir / "image_level_bbox_summary.csv", index=False)

    top100_records_df = pd.DataFrame(
        [{"label_name": k, "count": v} for k, v in label_counter_records.most_common(100)]
    )
    top100_records_df.to_csv(out_dir / "top100_labels_records.csv", index=False)

    top100_unique_df = pd.DataFrame(
        [{"label_name": k, "count": v} for k, v in label_counter_unique_top1.most_common(100)]
    )
    top100_unique_df.to_csv(out_dir / "top100_labels_unique_bbox_top1.csv", index=False)

    summary = {
        "n_images": int(len(image_df)),
        "record_count_per_image": {
            "min": int(image_df["record_count"].min()),
            "median": float(image_df["record_count"].median()),
            "mean": float(image_df["record_count"].mean()),
            "max": int(image_df["record_count"].max()),
        },
        "unique_bbox_count_per_image": {
            "min": int(image_df["unique_bbox_count"].min()),
            "median": float(image_df["unique_bbox_count"].median()),
            "mean": float(image_df["unique_bbox_count"].mean()),
            "max": int(image_df["unique_bbox_count"].max()),
        },
        "records_per_unique_bbox": {
            "min": float(image_df["records_per_unique_bbox"].min()),
            "median": float(image_df["records_per_unique_bbox"].median()),
            "mean": float(image_df["records_per_unique_bbox"].mean()),
            "max": float(image_df["records_per_unique_bbox"].max()),
        },
        "record_score_summary": {
            "mean": float(pd.Series(all_record_scores).mean()) if all_record_scores else None,
            "median": float(pd.Series(all_record_scores).median()) if all_record_scores else None,
            "max": float(max(all_record_scores)) if all_record_scores else None,
            "min": float(min(all_record_scores)) if all_record_scores else None,
        },
        "unique_bbox_top1_score_summary": {
            "mean": float(pd.Series(all_unique_bbox_top1_scores).mean()) if all_unique_bbox_top1_scores else None,
            "median": float(pd.Series(all_unique_bbox_top1_scores).median()) if all_unique_bbox_top1_scores else None,
            "max": float(max(all_unique_bbox_top1_scores)) if all_unique_bbox_top1_scores else None,
            "min": float(min(all_unique_bbox_top1_scores)) if all_unique_bbox_top1_scores else None,
        }
    }

    with open(out_dir / "summary_bbox_level.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # plots
    plot_hist(
        image_df["unique_bbox_count"].tolist(),
        xlabel="Unique bbox count per image",
        title="Distribution of unique bbox count per image",
        out_path=out_dir / "unique_bbox_per_image_hist.png",
        bins=50,
    )

    plot_hist(
        image_df["records_per_unique_bbox"].tolist(),
        xlabel="Records per unique bbox",
        title="Distribution of records per unique bbox",
        out_path=out_dir / "records_per_unique_bbox_hist.png",
        bins=50,
    )

    if all_unique_bbox_top1_scores:
        plot_hist(
            all_unique_bbox_top1_scores,
            xlabel="Top-1 score per unique bbox",
            title="Distribution of top-1 score per unique bbox",
            out_path=out_dir / "score_hist_unique_bbox_top1.png",
            bins=50,
        )

    if len(top100_records_df) > 0:
        plot_bar(
            top100_records_df["label_name"].astype(str).tolist(),
            top100_records_df["count"].tolist(),
            xlabel="Label name",
            title="Top 100 label frequencies (record-level)",
            out_path=out_dir / "top100_label_frequency_records.png",
        )

    if len(top100_unique_df) > 0:
        plot_bar(
            top100_unique_df["label_name"].astype(str).tolist(),
            top100_unique_df["count"].tolist(),
            xlabel="Label name",
            title="Top 100 label frequencies (unique-bbox top1)",
            out_path=out_dir / "top100_label_frequency_unique_bbox_top1.png",
        )

    report_lines = []
    report_lines.append("LLMDet BBox/Record Visualization Report")
    report_lines.append("=" * 80)
    report_lines.append(f"PRED_PKL: {pred_path}")
    report_lines.append(f"OUTPUT_DIR: {out_dir}")
    report_lines.append("")
    report_lines.append("[1] Record count per image")
    report_lines.append(str(summary["record_count_per_image"]))
    report_lines.append("")
    report_lines.append("[2] Unique bbox count per image")
    report_lines.append(str(summary["unique_bbox_count_per_image"]))
    report_lines.append("")
    report_lines.append("[3] Records per unique bbox")
    report_lines.append(str(summary["records_per_unique_bbox"]))
    report_lines.append("")
    report_lines.append("[4] Record score summary")
    report_lines.append(str(summary["record_score_summary"]))
    report_lines.append("")
    report_lines.append("[5] Unique bbox top1 score summary")
    report_lines.append(str(summary["unique_bbox_top1_score_summary"]))
    report_lines.append("")

    (out_dir / "bbox_record_visualization_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("Done.")
    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    main()