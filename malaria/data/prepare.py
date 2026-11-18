"""Stage 3: turn the unpacked images and annotations into a clean detection dataset.

What this stage does, in order:
  1. finds every image and every annotation per site and matches them by file name;
  2. drops images that cannot be used, and counts each reason (no annotation, unreadable,
     exact duplicate of an earlier image, duplicate file name);
  3. maps raw label spellings to the two classes, counts labels it does not recognise,
     clips boxes to the image and drops degenerate ones;
  4. splits by site: the training site into train / val / in-site test, the other site into a
     held-out cross-site test plus a small pool for the few-shot experiment;
  5. writes YOLO-format labels, split lists, dataset YAML files, an index and a data report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageOps

from malaria import config
from malaria.data.annotations import LabelMap, image_key, parse_tree

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIN_BOX_PX = 2.0


def clean_boxes(ann, label_map: LabelMap, width: int, height: int, stats: Counter, unknown: Counter):
    """Convert one image's raw boxes to (class, xmin, ymin, xmax, ymax) rows in pixels."""
    rows = []
    for box in ann.boxes:
        cls = label_map.resolve(box.label)
        if cls == "neg":
            stats["negative_labels"] += 1
            continue
        if cls is None:
            unknown[box.label] += 1
            continue
        x1, x2 = sorted((box.xmin, box.xmax))
        y1, y2 = sorted((box.ymin, box.ymax))
        if box.normalized:
            x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(float(width), x2), min(float(height), y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            stats["boxes_clipped"] += 1
        if cx2 - cx1 < MIN_BOX_PX or cy2 - cy1 < MIN_BOX_PX:
            stats["boxes_dropped_degenerate"] += 1
            continue
        rows.append((cls, cx1, cy1, cx2, cy2))
    return sorted(set(rows))


def yolo_lines(rows, width: int, height: int) -> list[str]:
    lines = []
    for cls, x1, y1, x2, y2 in rows:
        lines.append(
            f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return lines


def group_split(groups: dict[str, list[str]], fractions: dict[str, float], seed: int) -> dict[str, list[str]]:
    """Assign whole groups to splits. `fractions` maps split name -> share; the first split gets the rest."""
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    total = sum(len(groups[n]) for n in names)
    order = list(fractions)
    out = {name: [] for name in order}
    tail = order[1:]
    idx = 0
    for split in tail:
        target = fractions[split] * total
        while idx < len(names) and len(out[split]) < target:
            out[split].extend(groups[names[idx]])
            idx += 1
    for name in names[idx:]:
        out[order[0]].extend(groups[name])
    return {k: sorted(v) for k, v in out.items()}


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _place_upright(src: Path, dst: Path, orientation) -> tuple[int, int]:
    """Put the image at dst in the orientation a viewer shows it, and return (width, height).

    Phone photos often carry an EXIF rotation flag instead of rotated pixels. Annotation tools
    and OpenCV apply the flag, so labels refer to the rotated frame. Rotating the pixels once here,
    and dropping the flag, makes every later step see the same frame as the labels.
    """
    if orientation in (None, 1):
        _link(src, dst)
        with Image.open(dst) as im:
            return im.size
    if dst.exists():
        dst.unlink()  # may be a hard link to the raw file; never write through it
    with Image.open(src) as im:
        upright = ImageOps.exif_transpose(im)
        upright.save(dst, quality=95, subsampling=0)
        return upright.size


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_images(interim: Path, site: str, exclude: list[str]) -> list[Path]:
    """Images for one site: everything under its folder, plus images under `unknown/` whose path names the site."""
    found = []
    for base, need_name in ((interim / site, False), (interim / "unknown", True)):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
                continue
            rel = p.relative_to(base).as_posix()
            if any(re.search(pattern, rel) for pattern in exclude):
                continue
            if need_name and site.lower() not in rel.lower():
                continue
            found.append(p)
    return sorted(found)


def collect_site(site: str, interim: Path, out_dir: Path, label_map: LabelMap, seen_hashes: dict, report: dict, exclude: list[str]):
    """Process one site. Returns a list of index rows for the usable images."""
    stats, unknown = Counter(), Counter()
    anns, formats = parse_tree(interim / site) if (interim / site).exists() else ({}, {})
    if (interim / "unknown").exists():  # annotation files whose name did not say which site they belong to
        shared, shared_formats = parse_tree(interim / "unknown")
        for key, ann in shared.items():
            anns.setdefault(key, ann)
        formats = {k: formats.get(k, 0) + v for k, v in shared_formats.items()}
    images = find_images(interim, site, exclude)
    stats["images_found"] = len(images)
    stats["annotation_entries"] = len(anns)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    rows, used_keys = [], set()
    for path in images:
        key = image_key(path.name)
        if key in used_keys:
            stats["excluded_duplicate_name"] += 1
            continue
        if key not in anns:
            stats["excluded_no_annotation"] += 1
            continue
        try:
            with Image.open(path) as im:
                orientation = im.getexif().get(0x0112)
                im.verify()
            if orientation not in (None, 1):
                stats["images_rotated_upright"] += 1
        except Exception:
            stats["excluded_unreadable"] += 1
            continue
        digest = _md5(path)
        if digest in seen_hashes:
            stats["excluded_exact_duplicate"] += 1
            continue
        seen_hashes[digest] = f"{site}/{key}"
        used_keys.add(key)
        dst = out_dir / "images" / f"{key}{path.suffix.lower()}"
        width, height = _place_upright(path, dst, orientation)
        boxes = clean_boxes(anns[key], label_map, width, height, stats, unknown)
        (out_dir / "labels" / f"{key}.txt").write_text("\n".join(yolo_lines(boxes, width, height)))
        counts = Counter(cls for cls, *_ in boxes)
        sides = [max(x2 - x1, y2 - y1) for cls, x1, y1, x2, y2 in boxes if cls == 0]
        rows.append({
            "site": site, "key": key, "file": f"./{site}/images/{dst.name}", "width": width, "height": height,
            **{f"n_{name}": counts.get(i, 0) for i, name in enumerate(label_map.names)},
            "median_parasite_px": round(sorted(sides)[len(sides) // 2], 1) if sides else "",
        })
    stats["images_usable"] = len(rows)
    stats["orphan_annotations"] = len(set(anns) - {image_key(p.name) for p in images})
    report["sites"][site] = {
        "counts": dict(stats),
        "annotation_files": formats,
        "unknown_labels": dict(unknown.most_common(20)),
        "sample_file_names": [p.name for p in images[:8]],
    }
    return rows


def write_yaml(path: Path, root: Path, train: str, val: str, test: str | None, names: list[str]) -> None:
    doc = {"path": str(root.resolve()), "train": train, "val": val, "names": dict(enumerate(names))}
    if test:
        doc["test"] = test
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def run(cfg: config.Cfg) -> dict:
    interim, out = cfg.path("interim"), cfg.path("processed", make=True)
    split_cfg = cfg["split"]
    train_site, test_site = split_cfg["train_site"], split_cfg["test_site"]
    label_map = LabelMap(cfg["classes"])
    report = {"sites": {}, "splits": {}}
    seen_hashes: dict = {}
    rows = []
    for site in (train_site, test_site):  # training site first, so cross-site duplicates leave the test site
        if not (interim / site).exists() and not (interim / "unknown").exists():
            raise SystemExit(f"[prepare] {interim / site} does not exist; run the unpack stage first.")
        rows += collect_site(site, interim, out / site, label_map, seen_hashes, report, cfg["dataset"]["exclude"])
    by_site = defaultdict(list)
    for row in rows:
        by_site[row["site"]].append(row)
    for site in (train_site, test_site):
        if not by_site[site]:
            raise SystemExit(
                f"[prepare] no usable annotated images for site '{site}'. "
                f"See the counts below and reports/data_report.md.\n{json.dumps(report['sites'][site], indent=2)}"
            )

    regex = re.compile(split_cfg["group_regex"]) if split_cfg.get("group_regex") else None

    def groups_of(site_rows):
        groups = defaultdict(list)
        for row in site_rows:
            match = regex.search(row["key"]) if regex else None
            groups[match.group(1) if match else row["key"]].append(row["key"])
        return groups

    seed = split_cfg["seed"]
    train_groups = groups_of(by_site[train_site])
    in_site = group_split(
        train_groups,
        {"train": 0.0, "val": split_cfg["val_fraction"], "test": split_cfg["test_fraction"]},
        seed,
    )
    n_test_site = len(by_site[test_site])
    pool_n = min(split_cfg["adapt_pool"], n_test_site // 3)
    calib_n = min(split_cfg.get("calib_pool", 0), n_test_site // 3)
    cross_fracs = {"test": 0.0, "adapt": pool_n / max(n_test_site, 1)}
    if calib_n:
        cross_fracs["calib"] = calib_n / max(n_test_site, 1)
    cross = group_split(groups_of(by_site[test_site]), cross_fracs, seed)

    split_of = {}
    for name, keys in in_site.items():
        split_of.update({(train_site, k): f"{train_site}_{name}" for k in keys})
    for name, keys in cross.items():
        split_of.update({(test_site, k): f"{test_site}_{name}" for k in keys})
    file_of = {(r["site"], r["key"]): r["file"] for r in rows}

    def write_list(name: str, members: list[tuple[str, str]]) -> None:
        (out / f"{name}.txt").write_text("\n".join(file_of[m] for m in members) + "\n")
        report["splits"][name] = len(members)

    for name in ("train", "val", "test"):
        write_list(f"{train_site}_{name}", [(train_site, k) for k in in_site[name]])
    write_list(f"{test_site}_test", [(test_site, k) for k in cross["test"]])
    pool = list(cross["adapt"])
    random.Random(seed).shuffle(pool)
    write_list(f"{test_site}_adapt_pool", [(test_site, k) for k in sorted(pool)])
    sizes = [n for n in cfg["adapt"]["sizes"] if n <= len(pool)]
    if calib_n:  # labelled new-site images used only to choose a decision threshold there; never trained or tested on
        write_list(f"{test_site}_calib", [(test_site, k) for k in cross["calib"]])
    replay_pool = list(in_site["train"])
    random.Random(seed + 1).shuffle(replay_pool)
    ratio = float(cfg["adapt"].get("replay_ratio", 1.0))
    for n in sizes:  # nested subsets: the 50 images are inside the 100, which are inside the 200
        write_list(f"{test_site}_adapt_{n}", [(test_site, k) for k in sorted(pool[:n])])
        write_yaml(out / f"{test_site}_adapt_{n}.yaml", out, f"{test_site}_adapt_{n}.txt", f"{train_site}_val.txt", None, label_map.names)
        # Replay: the same new-site images plus training-site images, so the model keeps what it knew.
        old = sorted(replay_pool[: int(round(n * ratio))])
        write_list(f"{test_site}_adapt_{n}_replay", [(test_site, k) for k in sorted(pool[:n])] + [(train_site, k) for k in old])
        write_yaml(out / f"{test_site}_adapt_{n}_replay.yaml", out, f"{test_site}_adapt_{n}_replay.txt", f"{train_site}_val.txt", None, label_map.names)
    write_yaml(out / f"{train_site}.yaml", out, f"{train_site}_train.txt", f"{train_site}_val.txt", f"{train_site}_test.txt", label_map.names)

    for row in rows:
        row["split"] = split_of[(row["site"], row["key"])]
    with (out / "index.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report["adapt_sizes_available"] = sizes
    report["grouped_by_slide"] = bool(regex)
    report["class_names"] = label_map.names
    for site in (train_site, test_site):
        site_rows = by_site[site]
        sides = sorted(float(r["median_parasite_px"]) for r in site_rows if r["median_parasite_px"] != "")
        report["sites"][site]["summary"] = {
            "images": len(site_rows),
            **{f"boxes_{n}": sum(r[f"n_{n}"] for r in site_rows) for n in label_map.names},
            "images_without_parasites": sum(1 for r in site_rows if r[f"n_{label_map.names[0]}"] == 0),
            "median_parasite_box_px": sides[len(sides) // 2] if sides else None,
            "image_sizes": dict(Counter(f"{r['width']}x{r['height']}" for r in site_rows).most_common(5)),
        }
    reports = cfg.path("reports", make=True)
    (reports / "data_report.json").write_text(json.dumps(report, indent=2))
    (reports / "data_report.md").write_text(render_report(report))
    print(render_report(report))
    return report


def render_report(report: dict) -> str:
    lines = ["# Data report", "", "Written by `python -m malaria.data.prepare`. Every exclusion is counted here.", ""]
    for site, info in report["sites"].items():
        lines += [f"## {site}", "", "| Item | Count |", "| --- | ---: |"]
        lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in info["counts"].items()]
        lines += ["", f"Annotation files read: {info['annotation_files']}", ""]
        if info["counts"].get("images_rotated_upright"):
            lines += ["Images with an EXIF rotation flag were rotated to the orientation a viewer shows, "
                      "which is the frame their labels use, and the flag was dropped.", ""]
        if info["unknown_labels"]:
            lines += [f"Labels not mapped to a class (add them to `classes.aliases` if they should count): {info['unknown_labels']}", ""]
        if "summary" in info:
            lines += [f"Summary: {info['summary']}", ""]
        lines += [f"Sample file names: {info['sample_file_names']}", ""]
    lines += ["## Splits", "", "| Split | Images |", "| --- | ---: |"]
    lines += [f"| {k} | {v} |" for k, v in report["splits"].items()]
    lines += [""]
    if not report.get("grouped_by_slide"):
        lines += [
            "Images were split one by one. Several images can come from the same slide, so the in-site",
            "test score may be optimistic. Set `split.group_regex` to a pattern that captures the slide id",
            "in the file names above to keep each slide in one split. The cross-site test is unaffected.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    run(config.from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
