"""Stage: measure how different the two sites' images are, without using any labels.

Three checks on simple image statistics (brightness, contrast, saturation, sharpness, colour):
  * a two-sample Kolmogorov-Smirnov statistic per feature, training site vs test site;
  * a "can a classifier tell the sites apart" score (cross-validated AUROC; 0.5 = cannot);
  * the same two checks between two splits of the training site, as a control.
It also stores the training site's 1st-99th percentile range per feature. The API uses those
ranges to flag uploads that look unlike the training images.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from PIL import Image

from malaria import config
from malaria.models.evaluate import split_items
from service.quality import FEATURES, features


def feature_table(cfg: config.Cfg, split: str, rng: random.Random) -> np.ndarray:
    items = split_items(cfg, split)
    rng.shuffle(items)
    rows = []
    for it in items[: cfg["shift"]["max_images_per_site"]]:
        with Image.open(it["image"]) as img:
            f = features(img)
        rows.append([f[name] for name in FEATURES])
    return np.array(rows)


def compare(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    from scipy.stats import ks_2samp
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ks = {name: float(ks_2samp(a[:, i], b[:, i]).statistic) for i, name in enumerate(FEATURES)}
    x = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    folds = int(min(5, len(a), len(b)))
    auc = float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), x, y,
                                cv=folds, scoring="roc_auc").mean()) if folds >= 2 else None
    return {"ks": ks, "site_classifier_auroc": auc, "n": [int(len(a)), int(len(b))]}


def run(cfg: config.Cfg) -> dict:
    s = cfg["split"]
    rng = random.Random(s["seed"])
    train = feature_table(cfg, f"{s['train_site']}_train", rng)
    in_test = feature_table(cfg, f"{s['train_site']}_test", rng)
    cross = feature_table(cfg, f"{s['test_site']}_test", rng)
    lo, hi = np.percentile(train, 1, axis=0), np.percentile(train, 99, axis=0)
    result = {
        "features": list(FEATURES),
        "control_train_vs_in_site_test": compare(train, in_test, s["seed"]),
        "train_vs_cross_site_test": compare(train, cross, s["seed"]),
        "medians": {"train": dict(zip(FEATURES, np.median(train, 0).round(4).tolist())),
                    "cross_site_test": dict(zip(FEATURES, np.median(cross, 0).round(4).tolist()))},
        "quality_ranges": {name: [float(lo[i]), float(hi[i])] for i, name in enumerate(FEATURES)},
        "cross_site_images_outside_training_range": float(((cross < lo) | (cross > hi)).any(1).mean()),
        "in_site_images_outside_training_range": float(((in_test < lo) | (in_test > hi)).any(1).mean()),
    }
    out = cfg.path("reports", "metrics", "shift.json", make=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[shift] site-classifier AUROC: control {result['control_train_vs_in_site_test']['site_classifier_auroc']}, "
          f"cross-site {result['train_vs_cross_site_test']['site_classifier_auroc']}")
    return result


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    run(config.from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
