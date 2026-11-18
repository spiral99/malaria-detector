"""Stage 6: how many labelled images does a new site need, and what does adapting cost the old one?

Fine-tunes the selected training-site model on N labelled test-site images, for each N in
`adapt.sizes` and each mode in `adapt.modes`:
  plain   the N new-site images only (shows how much is forgotten about the training site);
  replay  the same N images plus N x replay_ratio training-site images (the standard remedy).
The new-site images come from a pool that is never part of the cross-site test set or the
calibration set. No test-site validation data is used: the schedule is fixed and the last
weights are kept.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from malaria import config, tracking
from malaria.models.train import _base_kwargs


def selected_model(cfg: config.Cfg) -> str:
    path = cfg.path("reports", "metrics", "detection.json")
    if not path.exists():
        raise SystemExit("[adapt] run the evaluate stage first: it selects the model to adapt.")
    return json.loads(path.read_text())["selected"]


def run(cfg: config.Cfg, force: bool = False) -> None:
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})
    site = cfg["split"]["test_site"]
    base = selected_model(cfg)
    a = cfg["adapt"]
    aug = dict(cfg["train"]["variants"].get(base) or {})
    for mode in a.get("modes", ["plain"]):
        for n in a["sizes"]:
            tag = f"adapt_{n}" if mode == "plain" else f"adapt_{n}_{mode}"
            data_yaml = cfg.path("processed", f"{site}_{tag}.yaml")
            out = cfg.path("models", f"{tag}.pt")
            if not data_yaml.exists():
                print(f"[adapt] skipped {tag}: the pool has fewer images than that")
                continue
            if out.exists() and not force:
                print(f"[adapt] kept {out.name}")
                continue
            with tracking.run(tag, {"base": base, "n_images": n, "mode": mode, "epochs": a["epochs"], "lr0": a["lr0"]}) as tracker:
                model = YOLO(str(cfg.path("models", f"{base}.pt")))
                model.train(data=str(data_yaml), epochs=a["epochs"], name=tag, val=False, optimizer="SGD",
                            lr0=a["lr0"], warmup_epochs=1, patience=0, **_base_kwargs(cfg), **aug)
                run_dir = Path(model.trainer.save_dir)
                shutil.copy2(run_dir / "weights" / "last.pt", out)
                tracker.artifact(run_dir / "results.csv")
            print(f"[adapt] wrote {out}")


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(config.from_args(args), args.force)


if __name__ == "__main__":
    main()
