import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


SPECULATIVE_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\bseems\b",
    r"\bappears\b",
    r"\blooks like\b",
    r"\blikely\b",
]

DETAIL_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "orange", "pink", "gray",
    "small", "large", "big", "tiny", "tall", "short", "young", "old",
    "wooden", "metal", "plastic", "glass", "brick",
    "left", "right", "top", "bottom", "front", "back", "near", "next", "beside", "behind",
    "under", "over", "inside", "outside", "middle", "center",
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "as", "and", "or", "but",
    "this", "that", "these", "those", "it", "its", "they", "them", "their", "there",
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def clean_caption_text(text: str) -> str:
    out = text
    for pattern in SPECULATIVE_PATTERNS:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out


def short_caption_vg(data: Dict) -> str:
    grounding = data.get("grounding", {})
    caption = grounding.get("caption", "")
    tags = data.get("tags", [])
    subsentence = [s.strip() for s in caption.split(".")]
    subsentence = [s for s in subsentence if s]
    subsentence = [s for s in subsentence if s not in tags]
    if not subsentence:
        return caption.strip()
    return ". ".join(subsentence) + "."


def pick_base_caption(data: Dict) -> str:
    conversations = data.get("conversations", [])
    if isinstance(conversations, list) and len(conversations) > 1 and isinstance(conversations[1], dict):
        value = conversations[1].get("value", "")
        if value:
            return value
    grounding = data.get("grounding", {})
    return grounding.get("caption", "")


@dataclass
class Variant:
    name: str
    clean_caption: bool
    use_short_cap: bool


def apply_variant_caption(data: Dict, dataset_mode: str, variant: Variant) -> str:
    caption = pick_base_caption(data)

    if dataset_mode == "VG" and variant.use_short_cap and "tags" in data:
        caption = short_caption_vg(data)

    if variant.clean_caption:
        cleaned = clean_caption_text(caption)
        if cleaned:
            caption = cleaned

    return caption


def count_speculative_hits(text: str) -> int:
    hits = 0
    for pattern in SPECULATIVE_PATTERNS:
        hits += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return hits


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_stats(
    files_with_mode: List[Tuple[str, str]],
    variants: List[Variant],
) -> List[Dict]:
    rows = []

    for variant in variants:
        caption_count = 0
        token_total = 0
        speculative_total = 0
        detail_token_total = 0
        noun_proxy_total = 0

        for jsonl_path, dataset_mode in files_with_mode:
            for data in iter_jsonl(jsonl_path):
                cap = apply_variant_caption(data, dataset_mode, variant)
                tokens = tokenize(cap)
                if not cap:
                    continue

                caption_count += 1
                token_total += len(tokens)
                speculative_total += count_speculative_hits(cap)

                detail_hits = sum(1 for t in tokens if t in DETAIL_WORDS or t.isdigit())
                detail_token_total += detail_hits

                tags = data.get("tags", [])
                noun_proxy_total += len(tags) if isinstance(tags, list) else 0

        avg_len = (token_total / caption_count) if caption_count else 0.0
        speculation_rate = (speculative_total / token_total) if token_total else 0.0
        detail_density = (detail_token_total / token_total) if token_total else 0.0
        noun_ratio = (noun_proxy_total / token_total) if token_total else 0.0

        rows.append(
            {
                "variant": variant.name,
                "caption_count": caption_count,
                "avg_token_length": round(avg_len, 4),
                "speculation_rate": round(speculation_rate, 6),
                "detail_density": round(detail_density, 6),
                "noun_count_proxy": noun_proxy_total,
                "noun_ratio_proxy": round(noun_ratio, 6),
                "clean_caption": variant.clean_caption,
                "use_short_cap": variant.use_short_cap,
            }
        )

    return rows


def save_csv(rows: List[Dict], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not rows:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Extension II caption statistics for 4 variants.")
    parser.add_argument(
        "--coco-jsonl",
        type=str,
        default="/autodl-fs/data/llmdet_extension/ann_variants/coco/strong/instances_train2017_ext.jsonl",
    )
    parser.add_argument(
        "--flickr-jsonl",
        type=str,
        default="/autodl-fs/data/llmdet_extension/ann_variants/flickr30k/strong/flickr_train_ext.jsonl",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="/autodl-fs/data/llmdet_extension/extension2_results/caption_stats.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    files_with_mode = [
        (args.coco_jsonl, "OD"),
        (args.flickr_jsonl, "VG"),
    ]

    for p, _ in files_with_mode:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input jsonl: {p}")

    variants = [
        Variant(name="raw-long", clean_caption=False, use_short_cap=False),
        Variant(name="cleaned-long", clean_caption=True, use_short_cap=False),
        Variant(name="raw-short", clean_caption=False, use_short_cap=True),
        Variant(name="cleaned-short", clean_caption=True, use_short_cap=True),
    ]

    rows = compute_stats(files_with_mode, variants)
    save_csv(rows, args.out_csv)

    print(f"Saved: {args.out_csv}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()