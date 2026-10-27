"""Quantitative EDA over the FIgLib annotations.

Answers the questions that actually drive preprocessing/training choices:
- How big are the smoke boxes in absolute pixels? (COCO small/medium/large)
- How big relative to the full image? (tells us what imgsz is sane)
- Where do boxes sit spatially, and what are their aspect ratios?

Outputs a JSON summary plus charts under docs/images/.

Usage:
    python scripts/analyze_dataset.py --raw data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
from prepare_figlib import list_csv_sequences, read_boxes  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

from PIL import Image  # noqa: E402


def classify_coco(area_px: float) -> str:
    if area_px < 32 * 32:
        return "small"
    if area_px < 96 * 96:
        return "medium"
    return "large"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="docs/images")
    args = parser.parse_args()

    from collections import Counter

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for seq in list_csv_sequences():
        boxes = read_boxes(seq)
        for name, bxs in boxes.items():
            img = raw / seq / name
            if not img.exists():
                continue
            with Image.open(img) as im:
                W, H = im.size
            for x1, y1, x2, y2 in bxs:
                bw, bh = x2 - x1, y2 - y1
                records.append(
                    {
                        "seq": seq,
                        "name": name,
                        "img_w": W,
                        "img_h": H,
                        "box_w": bw,
                        "box_h": bh,
                        "area": bw * bh,
                        "rel_w": bw / W,
                        "rel_h": bh / H,
                        "rel_area": (bw * bh) / (W * H),
                        "aspect": bw / bh,
                        "cx_rel": (x1 + x2) / 2 / W,
                        "cy_rel": (y1 + y2) / 2 / H,
                    }
                )

    arr = {
        k: np.array([r[k] for r in records], dtype=float)
        for k in records[0]
        if k not in ("seq", "name")
    }
    cats = [classify_coco(r["area"]) for r in records]
    cat_count = Counter(cats)

    # ---- numeric summary ----------------------------------------------
    summary = {
        "boxes": len(records),
        "sequences": len(list_csv_sequences()),
        "size_classes": dict(cat_count),
        "size_class_share": {k: round(v / len(records), 3) for k, v in cat_count.items()},
        "box_width_px": {
            "min": int(arr["box_w"].min()),
            "median": round(float(np.median(arr["box_w"])), 1),
            "max": int(arr["box_w"].max()),
        },
        "box_height_px": {
            "min": int(arr["box_h"].min()),
            "median": round(float(np.median(arr["box_h"])), 1),
            "max": int(arr["box_h"].max()),
        },
        "relative_area": {
            "median": round(float(np.median(arr["rel_area"])), 5),
            "p90": round(float(np.percentile(arr["rel_area"], 90)), 5),
        },
        "rel_width_percent": {
            "median": round(float(np.median(arr["rel_w"])) * 100, 2),
            "p90": round(float(np.percentile(arr["rel_w"], 90)) * 100, 2),
        },
        "aspect_ratio": {"median": round(float(np.median(arr["aspect"])), 2)},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- charts -------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 1) width vs height scatter
    ax = axes[0, 0]
    ax.scatter(arr["box_w"], arr["box_h"], s=12, alpha=0.55, c="#d62728")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for area, label, c in ((32 * 32, "32px", "gray"), (96 * 96, "96px", "black")):
        xs = np.logspace(0, 4, 200)
        ax.plot(xs, area / xs, "--", color=c, lw=1, label=f"{label}²")
    ax.set_xlabel("框宽 (px, log)")
    ax.set_ylabel("框高 (px, log)")
    ax.set_title("目标框宽高分布（对角线=等面积）")
    ax.legend()

    # 2) size class share
    ax = axes[0, 1]
    order = ["small", "medium", "large"]
    vals = [cat_count.get(k, 0) for k in order]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    ax.bar(order, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}\n({v/len(records):.0%})", ha="center", va="bottom")
    ax.set_ylabel("框数量")
    ax.set_title("按 COCO 尺寸标准分类（小<32² / 中<96² / 大）")

    # 3) relative width histogram (drives imgsz choice)
    ax = axes[0, 2]
    rel_w_pct = arr["rel_w"] * 100
    ax.hist(rel_w_pct, bins=30, color="#1f77b4", edgecolor="white")
    ax.axvline(
        np.median(rel_w_pct), color="red", ls="--", label=f"中位 {np.median(rel_w_pct):.1f}%"
    )
    ax.set_xlabel("框宽 / 图宽 (%)")
    ax.set_ylabel("框数量")
    ax.set_title("目标相对宽度分布（决定 imgsz 下限）")
    ax.legend()

    # 4) center position heatmap
    ax = axes[1, 0]
    ax.hist2d(arr["cx_rel"], arr["cy_rel"], bins=(20, 15), cmap="YlOrRd")
    ax.invert_yaxis()
    ax.set_xlabel("归一化 x 中心")
    ax.set_ylabel("归一化 y 中心")
    ax.set_title("目标中心位置热力图")

    # 5) aspect ratio
    ax = axes[1, 1]
    ax.hist(np.clip(arr["aspect"], 0, 10), bins=30, color="#9467bd", edgecolor="white")
    ax.set_xlabel("宽高比 (w/h)")
    ax.set_ylabel("框数量")
    ax.set_title("目标长宽比分布")

    # 6) boxes per sequence
    ax = axes[1, 2]
    per_seq = Counter(r["seq"] for r in records)
    names = [s.split("_FIRE_")[-1] for s in per_seq]
    ax.barh(names, list(per_seq.values()), color="#8c564b")
    ax.set_xlabel("框数量")
    ax.set_title("每个火情序列的标注框数")

    fig.suptitle("FIgLib 烟雾标注量化分析", fontsize=15, y=1.0)
    fig.tight_layout()
    target = out / "eda_overview.png"
    fig.savefig(target, dpi=130, bbox_inches="tight")
    print(f"wrote {target}")

    (out.parent / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
