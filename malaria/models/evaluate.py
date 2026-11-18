"""Stage 5: evaluate every model in-site and cross-site.

Rules that keep the numbers honest:
  * each model's "home" threshold is chosen on the training site's validation split;
  * a "local" threshold for the new site is chosen on the new site's calibration split, a set of
    labelled images that is never trained on and never part of the new site's test set;
  * cross-site results are reported at both thresholds: "as shipped" (home threshold, what a clinic
    gets with no labels) and "recalibrated" (local threshold, what a clinic gets after labelling the
    calibration images). Ranking metrics (AP) do not depend on the threshold;
  * the model that is served is chosen on the training site's validation split, never on the new site;
  * the new site's test set is only ever scored; it never contains adaptation or calibration images.

A model name of the form "base@2048" means the trained model "base" run at a 2048-pixel input.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from malaria import config, evalcore, tracking

IOUS_COCO = [0.5 + 0.05 * i for i in range(10)]


def split_items(cfg: config.Cfg, split: str) -> list[dict]:
    """Rows of the index (key, image path, label path, size) for one split."""
    root = cfg.path("processed")
    with (root / "index.csv").open() as fh:
        index = {row["file"]: row for row in csv.DictReader(fh)}
    items = []
    for line in (root / f"{split}.txt").read_text().splitlines():
        if not line.strip():
            continue
        row = index[line.strip()]
        image = root / line.strip()[2:]
        items.append({"key": f"{row['site']}/{row['key']}", "image": image,
                      "label": image.parent.parent / "labels" / f"{image.stem}.txt",
                      "width": int(row["width"]), "height": int(row["height"])})
    return items


def load_gt(items: list[dict]) -> dict:
    gts = {}
    for it in items:
        rows = []
        text = it["label"].read_text().strip() if it["label"].exists() else ""
        for line in text.splitlines():
            cls, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * it["width"], float(cy) * it["height"], float(w) * it["width"], float(h) * it["height"]
            rows.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, int(cls)])
        gts[it["key"]] = np.array(rows, dtype=float).reshape(-1, 5)
    return gts


def predict_split(cfg: config.Cfg, model_name: str, split: str, items: list[dict]) -> dict:
    """Predictions for one model on one split, cached on disk and refreshed when the weights or data change."""
    weights_name, _, size = model_name.partition("@")
    imgsz = int(size) if size else cfg["train"]["imgsz"]
    weights = cfg.path("models", f"{weights_name}.pt")
    cache = cfg.path("reports", "predictions", f"{model_name.replace('@', '_at')}__{split}.npz", make=True)
    stamp = weights.stat().st_mtime + cfg.path("processed", "index.csv").stat().st_mtime  # weights or data changed -> recompute
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if float(data["__stamp__"]) == stamp and int(data["__n__"]) == len(items):
            return {k: data[k] for k in data.files if not k.startswith("__")}
    from ultralytics import YOLO

    model = YOLO(str(weights))
    e, t = cfg["eval"], cfg["train"]
    kwargs = dict(imgsz=imgsz, conf=e["min_conf"], iou=e["nms_iou"], max_det=1000, verbose=False)
    if t.get("device") is not None:
        kwargs["device"] = t["device"]
    preds = {}
    for start in range(0, len(items), 16):
        chunk = items[start:start + 16]
        for it, res in zip(chunk, model.predict([str(i["image"]) for i in chunk], **kwargs)):
            b = res.boxes
            preds[it["key"]] = np.concatenate(
                [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], axis=1
            ).astype(np.float32).reshape(-1, 6)
    np.savez_compressed(cache, __stamp__=stamp, __n__=len(items), **preds)
    return preds


def score(preds: dict, gts: dict, names: list[str], thr: float, e: dict, seed: int) -> dict:
    out = {"n_images": len(gts), "classes": {}}
    for cls, name in enumerate(names):
        c = evalcore.collect(preds, gts, cls, e["iou"])
        row = {"n_boxes": int(c["n_gt"].sum()), "ap50": evalcore.average_precision(c)}
        row["ap50_95"] = float(np.nanmean([evalcore.average_precision(evalcore.collect(preds, gts, cls, i)) for i in IOUS_COCO]))
        row.update(evalcore.at_threshold(c, thr))
        row["ci95"] = evalcore.bootstrap(c, thr, e["bootstrap"], seed)
        out["classes"][name] = row
    return out


def wbc_confusion(preds: dict, gts: dict, thr: float, iou_thr: float) -> dict:
    """Of the parasite false positives, how many sit on a labelled white blood cell?"""
    fp_total = fp_on_wbc = 0
    for key, gt in gts.items():
        pred = preds.get(key, np.zeros((0, 6)))
        pred = pred[(pred[:, 5] == 0) & (pred[:, 4] >= thr)]
        flags, _ = evalcore.match_image(pred, gt[gt[:, 4] == 0], iou_thr)
        fps = pred[~flags]
        fp_total += len(fps)
        wbc = gt[gt[:, 4] == 1]
        if len(fps) and len(wbc):
            fp_on_wbc += int((evalcore.iou_matrix(fps[:, :4], wbc[:, :4]).max(1) >= 0.3).sum())
    return {"parasite_false_positives": fp_total, "on_a_labelled_wbc": fp_on_wbc}


def draw_worst(cfg: config.Cfg, split: str, items: list[dict], preds: dict, gts: dict, thr: float) -> list[str]:
    """Save the images with the most parasite errors: missed boxes yellow, false alarms red, hits green."""
    e = cfg["eval"]
    ranked = []
    for it in items:
        gt = gts[it["key"]]
        gt0 = gt[gt[:, 4] == 0]
        pred = preds.get(it["key"], np.zeros((0, 6)))
        pred = pred[(pred[:, 5] == 0) & (pred[:, 4] >= thr)]
        flags, matched = evalcore.match_image(pred, gt0, e["iou"])
        missed = np.ones(len(gt0), dtype=bool)
        missed[matched[matched >= 0]] = False
        ranked.append((int(missed.sum() + (~flags).sum()), it, gt0, pred, flags, missed))
    ranked.sort(key=lambda r: -r[0])
    out_dir = cfg.path("reports", "figures", "errors", make=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{split}_*.jpg"):
        old.unlink()
    saved = []
    for rank, (n_err, it, gt0, pred, flags, missed) in enumerate(ranked[: e["worst_images"]], 1):
        if n_err == 0:
            break
        img = Image.open(it["image"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        width = max(2, img.size[0] // 400)
        for box, hit in zip(pred, flags):
            draw.rectangle(box[:4].tolist(), outline=(0, 160, 60) if hit else (220, 40, 40), width=width)
        for box in gt0[missed]:
            draw.rectangle(box[:4].tolist(), outline=(250, 200, 0), width=width)
        img.thumbnail((1280, 1280))
        path = out_dir / f"{split}_{rank:02d}_{n_err}errors_{it['key'].split('/')[-1]}.jpg"
        img.save(path, quality=85)
        saved.append(path.name)
    return saved


def model_names(cfg: config.Cfg) -> list[str]:
    variants = [n for n in cfg["train"]["variants"] if cfg.path("models", f"{n}.pt").exists()]
    names = list(variants)
    names += [f"{v}@{size}" for v in variants for size in cfg["eval"].get("extra_imgsz", []) if size != cfg["train"]["imgsz"]]
    for mode in cfg["adapt"].get("modes", ["plain"]):
        for n in cfg["adapt"]["sizes"]:
            tag = f"adapt_{n}" if mode == "plain" else f"adapt_{n}_{mode}"
            if cfg.path("models", f"{tag}.pt").exists():
                names.append(tag)
    return names


def kind_of(name: str) -> str:
    if name.startswith("adapt_"):
        return "adapted"
    return "resized" if "@" in name else "trained"


def run(cfg: config.Cfg) -> dict:
    s, e = cfg["split"], cfg["eval"]
    names = cfg.class_names
    val, in_test, cross_test = f"{s['train_site']}_val", f"{s['train_site']}_test", f"{s['test_site']}_test"
    calib = f"{s['test_site']}_calib"
    has_calib = cfg.path("processed", f"{calib}.txt").exists()
    wanted = (val, in_test, cross_test) + ((calib,) if has_calib else ())
    splits = {name: split_items(cfg, name) for name in wanted}
    gts = {name: load_gt(items) for name, items in splits.items()}
    models = model_names(cfg)
    if not models:
        raise SystemExit("[evaluate] no trained models in the models folder; run the train stage first.")
    sizes = np.concatenate([np.maximum(g[g[:, 4] == 0][:, 2] - g[g[:, 4] == 0][:, 0], g[g[:, 4] == 0][:, 3] - g[g[:, 4] == 0][:, 1])
                            for g in gts[val].values()] or [np.zeros(0)])
    edges = [round(float(v), 1) for v in np.percentile(sizes, [33.3, 66.7])] if len(sizes) else [16.0, 32.0]

    def scored(preds, gt, thr):
        out = score(preds, gt, names, thr, e, s["seed"])
        out["recall_by_size"] = evalcore.recall_by_size(preds, gt, 0, e["iou"], thr, edges)
        out["wbc_confusion"] = wbc_confusion(preds, gt, thr, e["iou"])
        return out

    result = {"splits": {"val": val, "in_site_test": in_test, "cross_site_test": cross_test,
                         "cross_site_calib": calib if has_calib else None},
              "iou": e["iou"], "size_edges_px": edges, "models": {}}
    for model in models:
        preds = {name: predict_split(cfg, model, name, items) for name, items in splits.items()}
        thr_home = evalcore.best_f1_threshold(evalcore.collect(preds[val], gts[val], 0, e["iou"]))
        thr_local = (evalcore.best_f1_threshold(evalcore.collect(preds[calib], gts[calib], 0, e["iou"]))
                     if has_calib else None)
        entry = {"kind": kind_of(model), "threshold": thr_home, "threshold_local": thr_local, "splits": {}}
        for name in (val, in_test, cross_test):
            entry["splits"][name] = scored(preds[name], gts[name], thr_home)
        if has_calib:
            entry["cross_site_recalibrated"] = scored(preds[cross_test], gts[cross_test], thr_local)
        result["models"][model] = entry
        row = lambda name: entry["splits"][name]["classes"][names[0]]
        local = f"  local thr={thr_local:.2f} recall {entry['cross_site_recalibrated']['classes'][names[0]]['recall']:.3f}" if has_calib else ""
        print(f"[evaluate] {model:<20} home thr={thr_home:.2f}  parasite AP50 in-site {row(in_test)['ap50']:.3f}  "
              f"cross-site {row(cross_test)['ap50']:.3f}{local}")

    variants = [m for m in models if result["models"][m]["kind"] == "trained"]
    result["selected"] = max(variants, key=lambda m: result["models"][m]["splits"][val]["classes"][names[0]]["ap50"])
    sel = result["selected"]
    sel_entry = result["models"][sel]
    sel_preds = {name: predict_split(cfg, sel, name, splits[name]) for name in (in_test, cross_test)}
    cross_thr = sel_entry["threshold_local"] if has_calib else sel_entry["threshold"]
    result["worst_images"] = {
        in_test: draw_worst(cfg, in_test, splits[in_test], sel_preds[in_test], gts[in_test], sel_entry["threshold"]),
        cross_test: draw_worst(cfg, cross_test, splits[cross_test], sel_preds[cross_test], gts[cross_test], cross_thr),
    }
    result["worst_images_thresholds"] = {in_test: sel_entry["threshold"], cross_test: cross_thr}
    out = cfg.path("reports", "metrics", "detection.json", make=True)
    out.write_text(json.dumps(result, indent=2))
    with tracking.run("evaluate", {"selected": sel}) as tracker:
        for model, entry in result["models"].items():
            key = model.replace("@", "_at")
            tracker.metrics({f"{key}_in_site_ap50": entry["splits"][in_test]["classes"][names[0]]["ap50"],
                             f"{key}_cross_site_ap50": entry["splits"][cross_test]["classes"][names[0]]["ap50"]})
            if has_calib:
                tracker.metrics({f"{key}_cross_site_recalibrated_recall": entry["cross_site_recalibrated"]["classes"][names[0]]["recall"]})
        tracker.artifact(out)
    print(f"[evaluate] selected on {val}: {sel}; wrote {out}")
    return result


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    run(config.from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
