"""Stage 4: fine-tune a pretrained YOLO detector on the training site only.

One run per entry in `train.variants`. Each run starts from the same pretrained weights and
never sees an image from the test site. Best weights (by validation on the training site)
are copied to models/<variant>.pt.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from malaria import config, tracking


def _base_kwargs(cfg: config.Cfg) -> dict:
    t = cfg["train"]
    kwargs = dict(imgsz=t["imgsz"], batch=t["batch"], workers=t["workers"], seed=cfg["split"]["seed"],
                  project=str(cfg.path("runs")), exist_ok=True, verbose=False)
    if t.get("device") is not None:
        kwargs["device"] = t["device"]
    return kwargs


def train_variant(cfg: config.Cfg, name: str, aug: dict, force: bool = False) -> Path:
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})  # we log to MLflow ourselves, once per run
    out = cfg.path("models", f"{name}.pt", make=True)
    if out.exists() and not force:
        print(f"[train] kept {out.name} (use --force to retrain)")
        return out
    t = cfg["train"]
    data_yaml = cfg.path("processed", f"{cfg['split']['train_site']}.yaml")
    params = {"variant": name, "weights": t["weights"], "epochs": t["epochs"], "imgsz": t["imgsz"], "batch": t["batch"], **aug}
    with tracking.run(f"train-{name}", params) as tracker:
        model = YOLO(t["weights"])
        model.train(data=str(data_yaml), epochs=t["epochs"], patience=t["patience"], name=name, **_base_kwargs(cfg), **aug)
        run_dir = Path(model.trainer.save_dir)
        shutil.copy2(run_dir / "weights" / "best.pt", out)
        metrics = getattr(model.trainer, "metrics", {}) or {}
        tracker.metrics({k.replace("(B)", "").replace("/", "_"): v for k, v in metrics.items()}, prefix="val_")
        tracker.artifact(run_dir / "results.csv")
        tracker.artifact(run_dir / "results.png")
    print(f"[train] wrote {out}")
    return out


def run(cfg: config.Cfg, force: bool = False) -> None:
    for name, aug in cfg["train"]["variants"].items():
        train_variant(cfg, name, dict(aug or {}), force)


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(config.from_args(args), args.force)


if __name__ == "__main__":
    main()
