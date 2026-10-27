"""Temporal + per-camera quantitative analysis of the FIgLib annotations.

The single-image EDA told us *where* the boxes are; this one asks the question
that actually matters for a fire-alarm system: *how does the plume grow over
time, and how small is the earliest detectable box?*

Outputs charts under docs/images/ and a JSON summary.

Usage:
    python scripts/analyze_temporal.py --raw data/raw
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
from prepare_figlib import (  # noqa: E402
    camera_family,
    list_csv_sequences,
    read_boxes,
    stratified_val_sequences,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

from PIL import Image  # noqa: E402

OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$", re.IGNORECASE)
CAM_COLOR = {"mobo-c": "#d62728", "iqeye": "#1f77b4", "tcs3": "#2ca02c", "other": "#7f7f7f"}


def offset_of(name: str) -> int:
    m = OFFSET_RE.search(name)
    return int(m.group(1)) if m else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="docs/images")
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sequences = list_csv_sequences()
    val_seqs = stratified_val_sequences(sequences, 1, 42)

    # records: one per box, with offset + camera + split
    rows = []
    for seq in sequences:
        fam = camera_family(seq)
        split = "val" if seq in val_seqs else "train"
        for name, boxes in read_boxes(seq).items():
            img = raw / seq / name
            if not img.exists():
                continue
            with Image.open(img) as im:
                W, H = im.size
            off = offset_of(name)
            for x1, y1, x2, y2 in boxes:
                bw, bh = x2 - x1, y2 - y1
                rows.append(
                    {
                        "seq": seq,
                        "cam": fam,
                        "split": split,
                        "offset": off,
                        "box_w": bw,
                        "box_h": bh,
                        "area": bw * bh,
                        "rel_w": bw / W,
                        "img_w": W,
                    }
                )

    area = np.array([r["area"] for r in rows])
    offs = np.array([r["offset"] for r in rows])
    rel_w = np.array([r["rel_w"] for r in rows])

    # ---- first annotated box per sequence (== plume-appearance frame) --
    first_area_by_seq = {}
    for r in rows:
        if r["seq"] not in first_area_by_seq or r["offset"] < first_area_by_seq[r["seq"]]["offset"]:
            first_area_by_seq[r["seq"]] = r
    first_areas = np.array([v["area"] for v in first_area_by_seq.values()])
    first_relw = np.array([v["rel_w"] for v in first_area_by_seq.values()])

    summary = {
        "boxes": len(rows),
        "sequences": len(sequences),
        "annotation_start": "annotations begin at T+0 (plume-appearance frame) by design",
        "first_box_area_px2": {
            "median": round(float(np.median(first_areas)), 1),
            "min": int(first_areas.min()),
            "max": int(first_areas.max()),
            "per_seq": {k: int(v["area"]) for k, v in sorted(first_area_by_seq.items())},
        },
        "first_box_rel_width_pct": {
            "median": round(float(np.median(first_relw)) * 100, 2),
            "min": round(float(first_relw.min()) * 100, 2),
        },
        "area_growth": {
            "t_0_15min": round(float(np.median(area[(offs >= 0) & (offs <= 900)])), 1),
            "t_15_30min": round(float(np.median(area[(offs > 900) & (offs <= 1800)])), 1),
            "t_30_45min": round(float(np.median(area[(offs > 1800) & (offs <= 2700)])), 1),
            "t_gt_45min": round(float(np.median(area[offs > 2700])), 1),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- charts -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1) area vs offset, colored by camera
    ax = axes[0, 0]
    for fam in ("mobo-c", "iqeye", "tcs3"):
        mask = np.array([r["cam"] == fam for r in rows])
        ax.scatter(offs[mask] / 60, area[mask], s=14, alpha=0.55, color=CAM_COLOR[fam], label=fam)
    # binned median growth curve
    bins = np.arange(-40, 45, 5) * 60
    bin_c = (bins[:-1] + bins[1:]) / 2 / 60
    med = [
        np.median(area[(offs >= lo) & (offs < hi)])
        for lo, hi in zip(bins[:-1], bins[1:], strict=False)
    ]
    ax.plot(bin_c, med, "k-", lw=2, label="分箱中位数")
    ax.set_yscale("log")
    ax.set_xlabel("相对烟羽出现的时间 (min)")
    ax.set_ylabel("框面积 (px², log)")
    ax.set_title("目标面积随时间的演进（按相机着色）")
    ax.legend()

    # 2) first-annotated box area histogram
    ax = axes[0, 1]
    ax.hist(first_areas, bins=12, color="#1f77b4", edgecolor="white")
    ax.axvline(
        np.median(first_areas), color="red", ls="--", label=f"中位 {np.median(first_areas):.0f} px²"
    )
    ax.set_xlabel("首帧（T+0）标注框面积 (px²)")
    ax.set_ylabel("序列数")
    ax.set_title("每场火：最早可标注烟柱的尺寸")
    ax.legend()

    # 3) box width by camera family
    ax = axes[1, 0]
    fams = ["mobo-c", "iqeye", "tcs3"]
    data = [[r["box_w"] for r in rows if r["cam"] == f] for f in fams]
    bp = ax.boxplot(data, patch_artist=True)
    for patch, f in zip(bp["boxes"], fams, strict=False):
        patch.set_facecolor(CAM_COLOR[f])
        patch.set_alpha(0.55)
    ax.set_xticklabels(fams)
    ax.set_yscale("log")
    ax.set_ylabel("框宽 (px, log)")
    ax.set_title("不同相机型号的目标宽度分布")

    # 4) rel width train vs val
    ax = axes[1, 1]
    for split, color in (("train", "#d62728"), ("val", "#2ca02c")):
        m = np.array([r["split"] == split for r in rows])
        ax.hist(rel_w[m] * 100, bins=30, alpha=0.55, color=color, label=f"{split} (n={m.sum()})")
    ax.set_xlabel("框宽 / 图宽 (%)")
    ax.set_ylabel("框数量")
    ax.set_title("训练集 vs 验证集 目标相对宽度分布")
    ax.legend()

    fig.suptitle("FIgLib 时序与相机维度量化分析", fontsize=15, y=1.0)
    fig.tight_layout()
    target = out / "eda_temporal.png"
    fig.savefig(target, dpi=130, bbox_inches="tight")
    print(f"wrote {target}")

    (out.parent / "eda_temporal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
