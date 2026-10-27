"""Pseudo-labeling pipeline for the FIgLib smoke detector.

Motivation
----------
FIgLib ships 525 fire sequences but only 9 carry bounding boxes (309 boxes in
total). A detector trained on 9 sequences hits a *data* ceiling long before it
hits a model ceiling, so the highest-ROI next step is self-training: let the
current best model label the remaining sequences, keep only the confident
predictions, and fold them back into training.

Stages (each one writes to disk, so any stage can be re-run on its own)
----------------------------------------------------------------------
probe    score every unannotated sequence with its most mature frame;
         "unannotated" does not mean "has visible smoke"
collect  sample frames from selected sequences, download, resize
infer    run the teacher once at a very low conf, dump *raw* predictions
select   threshold those predictions, emit YOLO labels + a threshold sweep
audit    compare pseudo box sizes against ground-truth box sizes
merge    build a clean training tree (GT + pseudo); val stays GT-only

Why infer and select are separate: inference is the expensive step. Dumping
raw predictions once means the threshold sweep is free and can be redone
without touching the GPU.

Leakage guards
--------------
* every annotated sequence is excluded, which also removes the validation ones
* the merged validation split contains ground-truth frames only
* an entire sequence can be dropped when too few of its frames get a box
  (``--min-seq-ratio``): a real fire is visible in most frames, so a sparse
  sequence is usually the teacher hallucinating clouds

Data note
---------
Raw images stay in ``data/`` and are never committed: FIgLib is CC BY-NC-ND
4.0. Pseudo-labels are *derived* artefacts and carry the same restriction.

Usage
-----
    python scripts/pseudo_label.py collect --seqs-per-family 10 --frames 8
    python scripts/pseudo_label.py infer
    python scripts/pseudo_label.py select --conf 0.6
    python scripts/pseudo_label.py audit
    python scripts/pseudo_label.py merge
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_figlib import (  # noqa: E402
    BASE,
    camera_family,
    download_one,
    frame_offset,
    get_bytes,
    image_url,
    list_sequence_frames,
)

CONF_SWEEP = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70)
# COCO area thresholds in px² (our frames share one long side, so px² is comparable)
COCO_SMALL = 32**2
COCO_LARGE = 96**2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def list_all_sequences() -> list[str]:
    """Enumerate every sequence directory published on the FIgLib CDN."""
    html = get_bytes(f"{BASE}/index.html").decode("utf-8", "ignore")
    names = re.findall(r'href="?([^"\s>]+)/index\.html"?', html)
    return sorted({n for n in names if n and n != "Tar"})


def annotated_sequences(dataset: Path) -> tuple[set[str], set[str]]:
    """Recover (all annotated, validation) sequence names from an existing tree.

    Reading the filenames back is more reliable than re-running the split with
    the same seed, because it reflects what is actually on disk.
    """
    every: set[str] = set()
    val: set[str] = set()
    for split in ("train", "val"):
        for img in (dataset / "images" / split).glob("*.jpg"):
            seq = img.stem.split("__")[0]
            every.add(seq)
            if split == "val":
                val.add(seq)
    return every, val


def evenly_pick(items: list[str], count: int) -> list[str]:
    """Take `count` frames spread across the list.

    Random sampling clusters frames into the same minute, which makes the
    pseudo set redundant; even spacing covers the whole plume evolution.
    """
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[min(len(items) - 1, int(i * step))] for i in range(count)]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}; run the previous stage first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonlines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


# --------------------------------------------------------------------------
# stage 0: probe
# --------------------------------------------------------------------------
def stage_probe(args: argparse.Namespace) -> int:
    """Score every unannotated sequence with one late frame.

    "Unannotated sequence" does NOT mean "contains visible smoke": many FIgLib
    events are too far away, occluded, or simply not visible from that camera
    (confirmed by eye on several zero-hit sequences). Before investing in
    frames and labels, probe each sequence with its most mature frame -- the
    largest offset in the directory, where the plume has had the longest to
    grow -- and keep only the sequences the teacher reacts to.
    """
    import csv
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from PIL import Image

    dataset = Path(args.dataset)
    annotated, _ = annotated_sequences(dataset)
    pool = [s for s in list_all_sequences() if s not in annotated]
    print(f"[probe] candidate sequences: {len(pool)}")

    raw = Path(args.raw)
    img_dir = Path(args.pseudo) / "probe"
    img_dir.mkdir(parents=True, exist_ok=True)

    picked: list[tuple[str, str]] = []
    for seq in pool:
        try:
            frames = list_sequence_frames(seq)
        except OSError as exc:
            print(f"[probe]   skip {seq}: {exc}")
            continue
        post = [f for f in frames if (o := frame_offset(f)) is not None and o >= 0]
        if not post:
            continue
        picked.append((seq, max(post, key=frame_offset)))  # most mature plume
    print(f"[probe] sequences with post-ignition frames: {len(picked)}")

    def job(item: tuple[str, str]):
        seq, name = item
        url = image_url(seq, name)
        dst_raw = raw / seq / name
        ok, _note = download_one(url, dst_raw)[1], None
        if not ok:
            return seq, None
        dst = img_dir / f"probe__{seq}.jpg"
        try:
            with Image.open(dst_raw) as im:
                im = im.convert("RGB")
                w, h = im.size
                scale = args.long_side / max(w, h)
                if scale < 1.0:
                    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                im.save(dst, "JPEG", quality=95)
        except OSError:
            return seq, None
        return seq, dst

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(job, item) for item in picked]
        for i, future in enumerate(as_completed(futures), 1):
            if future.result()[1] is not None:
                ok += 1
            if i % 100 == 0 or i == len(futures):
                print(f"[probe]   {i}/{len(futures)} ({ok} ok)")

    from ultralytics import YOLO

    model = YOLO(args.teacher)
    rows = []
    for result in model.predict(
        source=str(img_dir),
        conf=0.01,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    ):
        seq = Path(result.path).stem.removeprefix("probe__")
        boxes = result.boxes
        conf = float(boxes.conf.max()) if boxes is not None and len(boxes) else 0.0
        rows.append(
            {
                "seq": seq,
                "family": camera_family(seq),
                "max_conf": round(conf, 4),
                "n_boxes": int(len(boxes)) if boxes is not None else 0,
            }
        )
    rows.sort(key=lambda r: (-r["max_conf"], r["seq"]))

    out = Path(args.pseudo) / "probe.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq", "family", "max_conf", "n_boxes"])
        writer.writeheader()
        writer.writerows(rows)

    hits = [r for r in rows if r["max_conf"] >= args.min_conf]
    print(f"[probe] scored {len(rows)} sequences, {len(hits)} react at conf>={args.min_conf}")
    print("[probe] top 10:")
    for row in rows[:10]:
        print(f"[probe]   {row['max_conf']:.3f}  {row['family']:8s} {row['seq']}")
    print(f"[probe] table -> {out}")
    return 0


# --------------------------------------------------------------------------
# stage 1: collect
# --------------------------------------------------------------------------
def stage_collect(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from PIL import Image

    dataset = Path(args.dataset)
    annotated, val_seqs = annotated_sequences(dataset)
    every = list_all_sequences()
    pool = [s for s in every if s not in annotated]
    print(f"[collect] sequences on CDN: {len(every)}")
    print(f"[collect] annotated (excluded): {len(annotated)} -> {sorted(annotated)}")
    print(f"[collect] val (excluded via annotated): {sorted(val_seqs)}")
    print(f"[collect] candidate pool: {len(pool)}")

    rng = random.Random(args.seed)
    if args.from_probe:
        import csv

        with open(args.from_probe, encoding="utf-8") as handle:
            cand = [
                row
                for row in csv.DictReader(handle)
                if float(row["max_conf"]) >= args.probe_min_conf
            ]
        cand.sort(key=lambda row: -float(row["max_conf"]))
        picked = [row["seq"] for row in cand[: args.probe_limit]]
        print(
            f"[collect] probe-driven: {len(cand)} sequences >= {args.probe_min_conf}, "
            f"taking top {len(picked)}"
        )
    else:
        grouped: dict[str, list[str]] = defaultdict(list)
        for seq in pool:
            grouped[camera_family(seq)].append(seq)
        picked = []
        for family, members in sorted(grouped.items()):
            rng.shuffle(members)
            take = members[: args.seqs_per_family]
            picked.extend(take)
            print(f"[collect]   {family:8s} available={len(members):3d} picked={len(take)}")

    raw = Path(args.raw)
    img_dir = Path(args.pseudo) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, int]] = []  # url, name, offset
    plan: list[dict] = []
    for seq in picked:
        try:
            frames = list_sequence_frames(seq)
        except OSError as exc:
            print(f"[collect]   skip {seq}: {exc}")
            continue
        pos = [f for f in frames if (o := frame_offset(f)) is not None and o >= 0]
        neg = [f for f in frames if (o := frame_offset(f)) is not None and o < 0]
        chosen_pos = evenly_pick(pos, args.frames)
        chosen_neg = evenly_pick(neg, args.neg_frames)
        for name, kind in [(n, "pos") for n in chosen_pos] + [(n, "neg") for n in chosen_neg]:
            tasks.append((image_url(seq, name), name, frame_offset(name) or 0))
            plan.append({"seq": seq, "name": name, "kind": kind, "offset": frame_offset(name) or 0})

    print(
        f"[collect] frames to fetch: {len(tasks)} "
        f"({sum(1 for p in plan if p['kind'] == 'pos')} pos / "
        f"{sum(1 for p in plan if p['kind'] == 'neg')} neg)"
    )

    fetch = [(url, raw / plan[i]["seq"] / name) for i, (url, name, _) in enumerate(tasks)]
    ok = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, url, path) for url, path in fetch]
        for i, future in enumerate(as_completed(futures), 1):
            _, success, _note = future.result()
            ok += success
            failed += not success
            if i % 50 == 0 or i == len(futures):
                print(f"[collect]   {i}/{len(futures)} ({ok} ok, {failed} failed)")

    meta: list[dict] = []
    for item in plan:
        src = raw / item["seq"] / item["name"]
        if not src.exists():
            continue
        stem = f"{item['seq']}__{Path(item['name']).stem}"
        dst = img_dir / f"{stem}.jpg"
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                scale = args.long_side / max(w, h)
                if scale < 1.0:
                    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                    w, h = im.size
                im.save(dst, "JPEG", quality=95)
        except OSError as exc:
            print(f"[collect]   skip {src.name}: {exc}")
            continue
        meta.append(
            {
                "image": f"{stem}.jpg",
                "seq": item["seq"],
                "kind": item["kind"],
                "offset": item["offset"],
                "width": w,
                "height": h,
            }
        )

    write_jsonlines(Path(args.pseudo) / "meta.jsonl", meta)
    pos_n = sum(1 for m in meta if m["kind"] == "pos")
    print(
        f"[collect] written {len(meta)} frames -> {img_dir} "
        f"({pos_n} pos / {len(meta) - pos_n} neg)"
    )
    return 0 if pos_n else 1


# --------------------------------------------------------------------------
# stage 2: infer
# --------------------------------------------------------------------------
def stage_infer(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    img_dir = Path(args.pseudo) / "images"
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {img_dir}; run collect first")

    print(f"[infer] teacher={args.teacher} images={len(images)} imgsz={args.imgsz}")
    model = YOLO(args.teacher)
    rows: list[dict] = []
    for i, result in enumerate(
        model.predict(
            source=str(img_dir),
            conf=0.01,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            stream=True,
            verbose=False,
        ),
        1,
    ):
        boxes = result.boxes
        preds = []
        if boxes is not None and len(boxes):
            xywhn = boxes.xywhn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (cx, cy, bw, bh), conf in zip(xywhn, confs, strict=False):
                preds.append([round(float(conf), 4), float(cx), float(cy), float(bw), float(bh)])
        rows.append({"image": Path(result.path).name, "preds": sorted(preds, key=lambda p: -p[0])})
        if i % 50 == 0 or i == len(images):
            print(f"[infer]   {i}/{len(images)}")

    out = Path(args.pseudo) / "preds.jsonl"
    write_jsonlines(out, rows)
    total = sum(len(r["preds"]) for r in rows)
    hit = sum(1 for r in rows if r["preds"])
    print(
        f"[infer] raw predictions: {total} boxes over {len(rows)} images "
        f"({hit} images with >=1 box) -> {out}"
    )
    return 0


# --------------------------------------------------------------------------
# stage 3: select
# --------------------------------------------------------------------------
def stage_select(args: argparse.Namespace) -> int:
    pseudo = Path(args.pseudo)
    meta = {m["image"]: m for m in read_jsonl(pseudo / "meta.jsonl")}
    rows = read_jsonl(pseudo / "preds.jsonl")

    # ---- threshold sweep (free: predictions are already on disk) ----------
    sweep = []
    for thr in CONF_SWEEP:
        boxes = sum(sum(1 for p in r["preds"] if p[0] >= thr) for r in rows)
        hits = sum(1 for r in rows if any(p[0] >= thr for p in r["preds"]))
        pos_rows = [r for r in rows if meta.get(r["image"], {}).get("kind") == "pos"]
        pos_hits = sum(1 for r in pos_rows if any(p[0] >= thr for p in r["preds"]))
        sweep.append(
            {
                "conf": thr,
                "boxes": boxes,
                "images_with_box": hits,
                "image_ratio": round(hits / len(rows), 4) if rows else 0.0,
                "pos_frames": len(pos_rows),
                "pos_hit_ratio": round(pos_hits / len(pos_rows), 4) if pos_rows else 0.0,
            }
        )

    print("[select] threshold sweep")
    print(f"  {'conf':>5} {'boxes':>7} {'imgs':>6} {'ratio':>7} {'pos_hit':>8}")
    for row in sweep:
        print(
            f"  {row['conf']:5.2f} {row['boxes']:7d} {row['images_with_box']:6d} "
            f"{row['image_ratio']:7.1%} {row['pos_hit_ratio']:8.1%}"
        )

    # ---- sequence-level guard --------------------------------------------
    # A hit-ratio cut (frames-with-box / frames) turned out to be wrong: the
    # first half of every sequence holds a plume too small for the teacher,
    # so even a good sequence scores ~20%. What a hallucinated sequence looks
    # like is *isolated* single-frame detections, so require a couple of
    # frames to fire instead of a ratio.
    per_seq: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        seq = meta.get(row["image"], {}).get("seq", "?")
        per_seq[seq].append(row)
    dropped = []
    for seq, members in per_seq.items():
        pos_members = [m for m in members if meta.get(m["image"], {}).get("kind") == "pos"]
        hits = sum(1 for m in pos_members if any(p[0] >= args.conf for p in m["preds"]))
        ratio = hits / len(pos_members) if pos_members else 1.0
        if hits < args.min_seq_hits or ratio < args.min_seq_ratio:
            dropped.append(seq)
    dropped.sort()
    if dropped:
        print(
            f"[select] dropping {len(dropped)} sequences "
            f"(fewer than {args.min_seq_hits} frame hits or ratio < "
            f"{args.min_seq_ratio:.0%}): {dropped[:6]}"
        )

    # ---- emit labels ------------------------------------------------------
    label_dir = pseudo / "labels"
    if label_dir.exists():
        shutil.rmtree(label_dir)
    label_dir.mkdir(parents=True)

    kept = 0
    kept_boxes = 0
    kept_seq: dict[str, dict] = {}
    for row in rows:
        info = meta.get(row["image"], {})
        seq = info.get("seq", "?")
        if seq in dropped:
            continue
        lines = [
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            for conf, cx, cy, bw, bh in row["preds"]
            if conf >= args.conf
        ]
        # A background frame (offset < 0) is a *true* negative: keep it with an
        # empty label file even though the teacher may hallucinate a box there.
        if not lines and info.get("kind") != "neg":
            continue
        (label_dir / f"{Path(row['image']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        kept += 1
        kept_boxes += len(lines)
        stat = kept_seq.setdefault(seq, {"frames": 0, "boxes": 0})
        stat["frames"] += 1
        stat["boxes"] += len(lines)

    report = {
        "teacher": getattr(args, "teacher", "models/best.pt"),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "sweep": sweep,
        "sequences_used": len(kept_seq),
        "sequences_dropped": dropped,
        "frames_kept": kept,
        "boxes_kept": kept_boxes,
        "sweep_note": "pos_hit_ratio = frames after ignition that get at least one box",
    }
    (pseudo / "select_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[select] kept {kept} frames / {kept_boxes} boxes "
        f"from {len(kept_seq)} sequences -> {label_dir}"
    )
    print(f"[select] report -> {pseudo / 'select_report.json'}")
    return 0


# --------------------------------------------------------------------------
# stage 4: audit
# --------------------------------------------------------------------------
def _box_stats(pairs: list[tuple[float, float, int, int]]) -> dict:
    """pairs = [(rel_w, rel_h, abs_w, abs_h)], all boxes pooled."""
    import statistics

    if not pairs:
        return {"n": 0}
    area = [w * h for _, _, w, h in pairs]
    small = sum(1 for a in area if a < COCO_SMALL)
    large = sum(1 for a in area if a >= COCO_LARGE)
    medium = len(area) - small - large
    rel_w = sorted(p[0] for p in pairs)
    return {
        "n": len(pairs),
        "rel_w_median": round(statistics.median(rel_w), 4),
        "rel_w_p10": round(rel_w[int(0.10 * (len(rel_w) - 1))], 4),
        "rel_w_p90": round(rel_w[int(0.90 * (len(rel_w) - 1))], 4),
        "abs_w_median": round(statistics.median([p[2] for p in pairs]), 1),
        "small_pct": round(100 * small / len(pairs), 1),
        "medium_pct": round(100 * medium / len(pairs), 1),
        "large_pct": round(100 * large / len(pairs), 1),
    }


def stage_audit(args: argparse.Namespace) -> int:
    from PIL import Image

    def collect(label_dir: Path, image_dir: Path) -> list[tuple[float, float, int, int]]:
        sizes: dict[str, tuple[int, int]] = {}
        pairs = []
        for txt in sorted(label_dir.glob("*.txt")):
            img = image_dir / f"{txt.stem}.jpg"
            if not img.exists():
                continue
            if txt.stem not in sizes:
                with Image.open(img) as im:
                    sizes[txt.stem] = im.size
            w, h = sizes[txt.stem]
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                bw, bh = float(parts[3]), float(parts[4])
                pairs.append((bw, bh, round(bw * w), round(bh * h)))
        return pairs

    dataset = Path(args.dataset)
    gt = collect(dataset / "labels" / "train", dataset / "images" / "train")
    ps = collect(Path(args.pseudo) / "labels", Path(args.pseudo) / "images")
    gt_stats, ps_stats = _box_stats(gt), _box_stats(ps)

    print("[audit] ground truth (train split) vs pseudo labels")
    keys = [
        "n",
        "rel_w_median",
        "rel_w_p10",
        "rel_w_p90",
        "abs_w_median",
        "small_pct",
        "medium_pct",
        "large_pct",
    ]
    print(f"  {'metric':<16} {'GT':>10} {'pseudo':>10}")
    for key in keys:
        print(
            f"  {key:<16} {str(gt_stats.get(key, '-')):>10} " f"{str(ps_stats.get(key, '-')):>10}"
        )

    if ps_stats["n"] and gt_stats["n"]:
        shift = ps_stats["rel_w_median"] - gt_stats["rel_w_median"]
        small_gap = ps_stats["small_pct"] - gt_stats["small_pct"]
        print(
            f"[audit] median relative width: {shift:+.2%} "
            f"(pseudo {'wider' if shift > 0 else 'narrower'} than GT)"
        )
        print(f"[audit] small-box share: {small_gap:+.1f} pp")
        if small_gap < -5:
            print(
                "[audit] WARNING: pseudo set under-represents small plumes — "
                "the teacher's own blind spot is being copied into training."
            )

    out = Path(args.pseudo) / "audit.json"
    out.write_text(
        json.dumps({"ground_truth": gt_stats, "pseudo": ps_stats}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[audit] report -> {out}")
    return 0


# --------------------------------------------------------------------------
# stage 5: merge
# --------------------------------------------------------------------------
def stage_merge(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    pseudo = Path(args.pseudo)
    out = Path(args.out)

    if not list((pseudo / "labels").glob("*.txt")):
        raise SystemExit(f"no pseudo labels in {pseudo / 'labels'}; run select first")

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    def link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        try:
            dst.hardlink_to(src)
        except (OSError, AttributeError):
            shutil.copy2(src, dst)

    # validation: ground truth only, never pseudo
    val_imgs = sorted((dataset / "images" / "val").glob("*.jpg"))
    for img in val_imgs:
        link_or_copy(img, out / "images" / "val" / img.name)
        label = dataset / "labels" / "val" / f"{img.stem}.txt"
        if label.exists():
            link_or_copy(label, out / "labels" / "val" / label.name)

    # training: ground truth + pseudo
    gt_imgs = sorted((dataset / "images" / "train").glob("*.jpg"))
    for img in gt_imgs:
        link_or_copy(img, out / "images" / "train" / img.name)
        label = dataset / "labels" / "train" / f"{img.stem}.txt"
        if label.exists():
            link_or_copy(label, out / "labels" / "train" / label.name)

    ps_imgs = sorted((pseudo / "images").glob("*.jpg"))
    added = 0
    added_boxes = 0
    for img in ps_imgs:
        label = pseudo / "labels" / f"{img.stem}.txt"
        if not label.exists():  # not selected in stage 3
            continue
        link_or_copy(img, out / "images" / "train" / img.name)
        link_or_copy(label, out / "labels" / "train" / label.name)
        added += 1
        added_boxes += len([ln for ln in label.read_text(encoding="utf-8").splitlines() if ln])

    total_train = len(list((out / "images" / "train").glob("*.jpg")))
    total_val = len(list((out / "images" / "val").glob("*.jpg")))

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "# FIgLib subset + pseudo labels -- generated by scripts/pseudo_label.py\n"
        "# GT = 9 annotated sequences; pseudo = teacher predictions on unannotated ones.\n"
        "# Pseudo labels inherit FIgLib's CC BY-NC-ND 4.0 terms; not for redistribution.\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: smoke\n",
        encoding="utf-8",
    )

    print(
        f"[merge] train: {len(gt_imgs)} GT + {added} pseudo = {total_train} images "
        f"({added_boxes} pseudo boxes)"
    )
    print(f"[merge] val  : {total_val} images (GT only)")
    print(f"[merge] data.yaml -> {yaml_path}")
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="data/yolo", help="existing GT dataset root")
    parser.add_argument("--pseudo", default="data/pseudo", help="pseudo-label workspace")
    parser.add_argument("--raw", default="data/pseudo/raw", help="raw download cache")
    parser.add_argument("--out", default="data/yolo_pseudo", help="merged dataset root")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("probe", help="score each unannotated sequence with one late frame")
    p.add_argument("--teacher", default="models/best.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="0")
    p.add_argument("--long-side", type=int, default=2048)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--min-conf",
        type=float,
        default=0.3,
        help="report how many sequences react at this confidence",
    )
    p.set_defaults(func=stage_probe)

    p = sub.add_parser("collect", help="sample and download unannotated frames")
    p.add_argument("--seqs-per-family", type=int, default=10)
    p.add_argument(
        "--from-probe",
        default=None,
        metavar="CSV",
        help="take sequences from a probe.csv instead of family sampling",
    )
    p.add_argument("--probe-min-conf", type=float, default=0.3)
    p.add_argument("--probe-limit", type=int, default=60)
    p.add_argument("--frames", type=int, default=8, help="post-ignition frames per sequence")
    p.add_argument("--neg-frames", type=int, default=2, help="pre-ignition frames per sequence")
    p.add_argument("--long-side", type=int, default=2048)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=stage_collect)

    p = sub.add_parser("infer", help="run the teacher, dump raw predictions")
    p.add_argument("--teacher", default="models/best.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.set_defaults(func=stage_infer)

    p = sub.add_parser("select", help="threshold raw predictions into labels")
    p.add_argument("--conf", type=float, default=0.6)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument(
        "--min-seq-hits",
        type=int,
        default=2,
        help="drop sequences with fewer than this many frames with a box",
    )
    p.add_argument(
        "--min-seq-ratio",
        type=float,
        default=0.0,
        help="optional: also require a minimum frame hit ratio",
    )
    p.set_defaults(func=stage_select)

    p = sub.add_parser("audit", help="compare pseudo vs GT box statistics")
    p.set_defaults(func=stage_audit)

    p = sub.add_parser("merge", help="build the merged training tree")
    p.set_defaults(func=stage_merge)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
