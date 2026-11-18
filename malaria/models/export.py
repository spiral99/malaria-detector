"""Stage: export the selected model to ONNX for CPU serving and write its metadata.

Also checks that the ONNX model, run through the serving code, agrees with the PyTorch model
on validation images, and copies a few sample photos for the demo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from malaria import config
from malaria.models.evaluate import predict_split, split_items


def run(cfg: config.Cfg, model_name: str | None = None) -> Path:
    from ultralytics import YOLO

    s, t = cfg["split"], cfg["train"]
    detection = json.loads(cfg.path("reports", "metrics", "detection.json").read_text())
    name = model_name or detection["selected"]
    entry = detection["models"][name]
    weights = cfg.path("models", f"{name}.pt")
    exported = Path(YOLO(str(weights)).export(format="onnx", imgsz=t["imgsz"], opset=12, simplify=False, dynamic=False, device="cpu"))
    onnx_path = cfg.path("models", "model.onnx")
    shutil.move(str(exported), onnx_path)

    shift_path = cfg.path("reports", "metrics", "shift.json")
    ranges = json.loads(shift_path.read_text())["quality_ranges"] if shift_path.exists() else {}
    cross = entry["splits"][detection["splits"]["cross_site_test"]]["classes"][cfg.class_names[0]]
    in_site = entry["splits"][detection["splits"]["in_site_test"]]["classes"][cfg.class_names[0]]
    meta = {
        "name": name, "classes": cfg.class_names, "imgsz": t["imgsz"], "conf_threshold": entry["threshold"],
        "nms_iou": cfg["eval"]["nms_iou"], "low_conf_margin": cfg["serve"]["low_conf_margin"],
        "trained_on": s["train_site"] if not name.startswith("adapt_") else f"{s['train_site']} + {name.split('_')[1]} {s['test_site']} images",
        "base_weights": t["weights"], "exported": dt.date.today().isoformat(),
        "quality_ranges": ranges,
        "metrics": {"in_site_ap50": in_site["ap50"], "cross_site_ap50": cross["ap50"],
                    "in_site_recall": in_site["recall"], "cross_site_recall_as_shipped": cross["recall"],
                    "cross_site_threshold_local": entry.get("threshold_local"),
                    "cross_site_recall_recalibrated": (entry.get("cross_site_recalibrated") or {}).get("classes", {})
                                                      .get(cfg.class_names[0], {}).get("recall")},
    }
    meta_path = cfg.path("models", "model_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    # Parity: the served ONNX path must count about the same objects as the evaluated PyTorch path.
    from service.inference import Detector

    det = Detector(onnx_path, meta_path)
    val = detection["splits"]["val"]
    items = split_items(cfg, val)
    torch_preds = predict_split(cfg, name, val, items)
    diffs = []
    for it in items[:12]:
        with Image.open(it["image"]) as img:
            rows = det.predict(img.convert("RGB"))
        ref = torch_preds[it["key"]]
        diffs.append(abs(int((rows[:, 5] == 0).sum()) - int(((ref[:, 5] == 0) & (ref[:, 4] >= det.threshold)).sum())))
    meta["onnx_parity"] = {"images": len(diffs), "mean_abs_parasite_count_difference": float(np.mean(diffs)) if diffs else None}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[export] {name} -> {onnx_path.name}; parity on {len(diffs)} images: mean count difference {meta['onnx_parity']['mean_abs_parasite_count_difference']}")

    samples = config.ROOT / "samples" if cfg.path("models") == config.ROOT / "models" else cfg.path("models").parent / "samples"
    samples.mkdir(exist_ok=True)
    for split in (detection["splits"]["in_site_test"], detection["splits"]["cross_site_test"]):
        for it in split_items(cfg, split)[:3]:
            shutil.copy2(it["image"], samples / f"{split}_{it['image'].name}")
    return onnx_path


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--model", default=None, help="model name to export (default: the one selected on validation)")
    args = parser.parse_args()
    run(config.from_args(args), args.model)


if __name__ == "__main__":
    main()
