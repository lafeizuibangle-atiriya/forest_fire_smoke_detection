"""Prepare the FIgLib subset into YOLO format.

Data source : HPWREN FIgLib (https://www.hpwren.ucsd.edu/FIgLib/)
License     : CC BY-NC-ND 4.0 -- raw data must NOT be redistributed.
Credit      : Data courtesy of HPWREN, UC San Diego.

What this script does
---------------------
1. Download the 9 officially annotated sequence CSVs (absolute xyxy boxes).
2. Download the annotated JPEGs, plus extra pre-ignition frames used as
   negative (no-smoke) samples.
3. Convert every annotation to normalized YOLO ``cls cx cy w h`` lines.
4. Split by *fire sequence* (never by image) so validation frames cannot
   leak from a sequence the model already saw.
5. Emit ``data.yaml`` for Ultralytics.

Usage
-----
    python scripts/prepare_figlib.py --neg-per-seq 30

The raw images land in ``data/raw`` and the YOLO tree in ``data/yolo``.
Both are gitignored on purpose: FIgLib forbids redistribution.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, unquote

BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
LABEL_CSV = f"{BASE}/Miscellaneous/Labels/HPWREN-BB/CSV"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$", re.IGNORECASE)
CITATION = (
    "Data courtesy of HPWREN, UC San Diego (https://www.hpwren.ucsd.edu/). "
    "FIgLib is CC BY-NC-ND 4.0; do not redistribute the raw images."
)


def get_bytes(url: str, timeout: int = 90) -> bytes:
    """Fetch a URL with a browser UA (the CDN rejects the default one)."""
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_csv_sequences() -> list[str]:
    """Return the annotated sequence names (CSV stems) published by HPWREN."""
    html = get_bytes(f"{LABEL_CSV}/index.html").decode("utf-8", "ignore")
    return sorted({n[: -len(".csv")] for n in re.findall(r"href=([^\s>]+\.csv)", html)})


def read_boxes(sequence: str) -> dict[str, list[tuple[int, int, int, int]]]:
    """Download one sequence CSV and group boxes by image filename."""
    text = get_bytes(f"{LABEL_CSV}/{sequence}.csv").decode("utf-8", "ignore")
    grouped: dict[str, list[tuple[int, int, int, int]]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        name = Path(row["Filename"]).name
        box = (
            int(row["MinX"]),
            int(row["MinY"]),
            int(row["MaxX"]),
            int(row["MaxY"]),
        )
        grouped.setdefault(name, []).append(box)
    return grouped


def list_sequence_frames(sequence: str) -> list[str]:
    """List every JPEG filename in a sequence directory.

    The CDN's autoindex percent-encodes the literal ``+`` in post-ignition
    filenames (``1465067760_%2B02160.jpg``). Decode it here so that
    :func:`frame_offset` sees a real ``+``; :func:`image_url` re-encodes it
    when building the request, so downloads keep working.
    """
    html = get_bytes(f"{BASE}/{sequence}/index.html").decode("utf-8", "ignore")
    names = re.findall(r"href=([^\s>]+\.jpg)", html, flags=re.IGNORECASE)
    return sorted({unquote(n) for n in names})


def frame_offset(name: str) -> int | None:
    """Seconds relative to the visible plume appearance, from the filename."""
    match = OFFSET_RE.search(name)
    return int(match.group(1)) if match else None


def image_url(sequence: str, name: str) -> str:
    """Build the image URL.

    FIgLib filenames encode the offset from plume appearance as ``+02340`` /
    ``-02400``. A literal ``+`` in a path segment is decoded as a space by the
    CDN and answered with 403, so it must be percent-encoded as ``%2B``.
    """
    return f"{BASE}/{sequence}/{quote(name)}"


def download_one(url: str, target: Path) -> tuple[str, bool, str]:
    """Download a single file unless it already exists."""
    if target.exists() and target.stat().st_size > 1024:
        return target.name, True, "cached"
    try:
        payload = get_bytes(url, timeout=120)
        if len(payload) < 1024:
            return target.name, False, "too small"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target.name, True, f"{len(payload) // 1024}KB"
    except (urllib.error.URLError, OSError) as exc:
        return target.name, False, str(exc)[:80]


def to_yolo_line(cls_id: int, box: tuple[int, int, int, int], w: int, h: int) -> str:
    """Convert an absolute xyxy box to a normalized YOLO label line."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
    y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def camera_family(sequence: str) -> str:
    """Group sequences by imager model.

    FIgLib mixes several camera models with very different optics and
    resolution. If the validation split ends up holding a model that barely
    appears in training, the metrics measure domain shift instead of skill,
    so the split is stratified by this family.
    """
    lowered = sequence.lower()
    for token in ("tcs3", "iqeye", "mobo-c", "mobo-m", "mobo"):
        if token in lowered:
            return token
    return "other"


def stratified_val_sequences(sequences: list[str], per_group: int, seed: int) -> set[str]:
    """Pick validation sequences so every camera family is represented."""
    grouped: dict[str, list[str]] = {}
    for seq in sequences:
        grouped.setdefault(camera_family(seq), []).append(seq)
    rng = random.Random(seed)
    picked: set[str] = set()
    for _family, members in sorted(grouped.items()):
        # A family needs at least two sequences: keep one for training.
        if len(members) < 2:
            continue
        rng.shuffle(members)
        picked.update(members[: min(per_group, len(members) - 1)])
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw", help="raw image cache")
    parser.add_argument("--out", default="data/yolo", help="YOLO dataset root")
    parser.add_argument(
        "--neg-per-seq", type=int, default=30, help="pre-ignition frames per sequence"
    )
    parser.add_argument(
        "--val-per-group", type=int, default=1, help="validation sequences per camera family"
    )
    parser.add_argument(
        "--long-side",
        type=int,
        default=2048,
        help="resize long side; 2048 keeps small plumes visible",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from PIL import Image

    raw = Path(args.raw)
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    sequences = list_csv_sequences()
    print(f"[1/5] annotated sequences: {len(sequences)}")

    # ---- annotations -------------------------------------------------
    boxes_by_seq: dict[str, dict[str, list[tuple[int, int, int, int]]]] = {}
    for seq in sequences:
        boxes_by_seq[seq] = read_boxes(seq)
        print(f"      {seq}: {sum(len(v) for v in boxes_by_seq[seq].values())} boxes")

    # ---- negative frames --------------------------------------------
    rng = random.Random(args.seed)
    negatives: dict[str, list[str]] = {}
    if args.neg_per_seq > 0:
        for seq in sequences:
            frames = list_sequence_frames(seq)
            pre = [f for f in frames if (o := frame_offset(f)) is not None and o < 0]
            if len(pre) > args.neg_per_seq:
                pre = rng.sample(pre, args.neg_per_seq)
            negatives[seq] = pre
        print(f"[2/5] negative (no-smoke) frames: {sum(len(v) for v in negatives.values())}")

    # ---- download ----------------------------------------------------
    tasks: list[tuple[str, Path, str]] = []
    for seq in sequences:
        for name in boxes_by_seq[seq]:
            tasks.append((image_url(seq, name), raw / seq / name, seq))
        for name in negatives.get(seq, []):
            tasks.append((image_url(seq, name), raw / seq / name, seq))

    print(f"[3/5] downloading {len(tasks)} images -> {raw}")
    ok = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, url, path) for url, path, _ in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            name, success, note = future.result()
            if success:
                ok += 1
            else:
                failed.append(f"{name}: {note}")
            if i % 50 == 0 or i == len(tasks):
                print(f"      {i}/{len(tasks)} done ({ok} ok, {len(failed)} failed)")
    if failed:
        print(f"      WARNING {len(failed)} failures, first 5: {failed[:5]}")

    # ---- convert + split ---------------------------------------------
    seqs = list(sequences)
    rng.shuffle(seqs)
    val_seqs = stratified_val_sequences(seqs, args.val_per_group, args.seed)
    print(f"[4/5] val sequences (stratified by camera): {sorted(val_seqs)}")
    for seq in sorted(sequences):
        print(f"      {camera_family(seq):8s} {seq} -> {'val' if seq in val_seqs else 'train'}")

    stats = {
        "train": {"img": 0, "pos": 0, "neg": 0, "box": 0},
        "val": {"img": 0, "pos": 0, "neg": 0, "box": 0},
    }

    for seq in sequences:
        split = "val" if seq in val_seqs else "train"
        items: list[tuple[str, list[tuple[int, int, int, int]]]] = []
        for name, boxes in boxes_by_seq[seq].items():
            items.append((name, boxes))
        for name in negatives.get(seq, []):
            items.append((name, []))

        for name, boxes in items:
            src = raw / seq / name
            if not src.exists():
                continue
            stem = f"{seq}__{Path(name).stem}"
            dst_img = out / "images" / split / f"{stem}.jpg"
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    scale = args.long_side / max(w, h)
                    if scale < 1.0:
                        im = im.resize(
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            Image.LANCZOS,
                        )
                    im.save(dst_img, "JPEG", quality=95)
            except OSError as exc:
                print(f"      skip {src.name}: {exc}")
                continue

            lines = [to_yolo_line(0, b, w, h) for b in boxes]
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

            stats[split]["img"] += 1
            stats[split]["box"] += len(lines)
            if lines:
                stats[split]["pos"] += 1
            else:
                stats[split]["neg"] += 1

    # ---- dataset yaml -------------------------------------------------
    yaml_path = out / "data.yaml"
    yaml_text = (
        "# FIgLib smoke subset -- generated by scripts/prepare_figlib.py\n"
        f"# {CITATION}\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: smoke\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("[5/5] summary")
    for split, s in stats.items():
        ratio = f"{s['pos'] / s['img']:.1%}" if s["img"] else "n/a"
        print(
            f"      {split:5s}: {s['img']:4d} images "
            f"({s['pos']} smoke / {s['neg']} background, {ratio} positive), "
            f"{s['box']} boxes"
        )
    print(f"      data.yaml -> {yaml_path}")
    print(f"NOTE: {CITATION}")
    return 0 if stats["train"]["img"] and stats["val"]["img"] else 1


if __name__ == "__main__":
    sys.exit(main())
