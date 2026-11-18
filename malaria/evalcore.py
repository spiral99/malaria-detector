"""Detection metrics in plain NumPy, so they can be unit-tested and bootstrapped.

Conventions: boxes are [x1, y1, x2, y2] in pixels. A prediction array has rows
[x1, y1, x2, y2, confidence, class]. A ground-truth array has rows [x1, y1, x2, y2, class].
A prediction is a true positive when it overlaps a not-yet-matched ground-truth box of the
same class by at least the IoU threshold; predictions are matched in order of confidence.
"""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_image(pred: np.ndarray, gt: np.ndarray, iou_thr: float):
    """Greedy matching for one image and one class.

    Returns (tp flags per prediction in the input order, index of the matched gt box or -1).
    """
    tp = np.zeros(len(pred), dtype=bool)
    matched_gt = np.full(len(pred), -1, dtype=int)
    if len(pred) == 0 or len(gt) == 0:
        return tp, matched_gt
    ious = iou_matrix(pred[:, :4], gt[:, :4])
    taken = np.zeros(len(gt), dtype=bool)
    for i in np.argsort(-pred[:, 4], kind="stable"):
        candidates = np.where(~taken, ious[i], -1.0)
        j = int(np.argmax(candidates))
        if candidates[j] >= iou_thr:
            tp[i], matched_gt[i], taken[j] = True, j, True
    return tp, matched_gt


def collect(preds: dict, gts: dict, cls: int, iou_thr: float) -> dict:
    """Match every image for one class and flatten the result for metric computation.

    `preds` and `gts` map image key -> array. Every image in `gts` is evaluated.
    """
    keys = sorted(gts)
    conf, tp, img = [], [], []
    n_gt = np.zeros(len(keys))
    for idx, key in enumerate(keys):
        gt = gts[key]
        gt = gt[gt[:, 4] == cls] if len(gt) else np.zeros((0, 5))
        pred = preds.get(key, np.zeros((0, 6)))
        pred = pred[pred[:, 5] == cls] if len(pred) else np.zeros((0, 6))
        flags, _ = match_image(pred, gt, iou_thr)
        conf.append(pred[:, 4])
        tp.append(flags)
        img.append(np.full(len(pred), idx))
        n_gt[idx] = len(gt)
    return {
        "keys": keys,
        "conf": np.concatenate(conf) if conf else np.zeros(0),
        "tp": np.concatenate(tp).astype(float) if tp else np.zeros(0),
        "img": np.concatenate(img).astype(int) if img else np.zeros(0, dtype=int),
        "n_gt": n_gt,
    }


def average_precision(c: dict, weights: np.ndarray | None = None) -> float:
    """COCO-style 101-point interpolated AP. `weights` gives each image a multiplicity (bootstrap)."""
    w = np.ones(len(c["n_gt"])) if weights is None else weights
    total_gt = float((c["n_gt"] * w).sum())
    if total_gt == 0:
        return float("nan")
    if len(c["conf"]) == 0:
        return 0.0
    order = np.argsort(-c["conf"], kind="stable")
    det_w = w[c["img"][order]]
    tp = np.cumsum(c["tp"][order] * det_w)
    fp = np.cumsum((1 - c["tp"][order]) * det_w)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    for i in range(len(precision) - 2, -1, -1):  # make precision non-increasing
        precision[i] = max(precision[i], precision[i + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    return float(np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0).mean())


def at_threshold(c: dict, thr: float, weights: np.ndarray | None = None) -> dict:
    """Precision, recall, F1 and image-level screening metrics at one confidence threshold."""
    w = np.ones(len(c["n_gt"])) if weights is None else weights
    keep = c["conf"] >= thr
    det_w = w[c["img"][keep]]
    tp = float((c["tp"][keep] * det_w).sum())
    fp = float(((1 - c["tp"][keep]) * det_w).sum())
    total_gt = float((c["n_gt"] * w).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / total_gt if total_gt else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and precision + recall else 0.0
    n_pred = np.bincount(c["img"][keep], minlength=len(c["n_gt"])).astype(float)
    has_gt, has_pred = c["n_gt"] > 0, n_pred > 0
    pos, neg = float((w * has_gt).sum()), float((w * ~has_gt).sum())
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "image_sensitivity": float((w * (has_gt & has_pred)).sum() / pos) if pos else float("nan"),
        "image_specificity": float((w * (~has_gt & ~has_pred)).sum() / neg) if neg else float("nan"),
        "count_mae": float((w * np.abs(n_pred - c["n_gt"])).sum() / max(w.sum(), 1e-9)),
        "count_bias": float((w * (n_pred - c["n_gt"])).sum() / max(w.sum(), 1e-9)),
    }


def best_f1_threshold(c: dict) -> float:
    """The confidence threshold that maximises F1. Call this on validation data only."""
    if len(c["conf"]) == 0:
        return 0.25
    candidates = np.unique(np.round(np.clip(c["conf"], 0.01, 0.99), 2))
    scores = [at_threshold(c, t)["f1"] for t in candidates]
    return round(float(candidates[int(np.argmax(scores))]), 2)


def bootstrap(c: dict, thr: float, n: int, seed: int = 0) -> dict:
    """95% intervals from resampling images with replacement."""
    rng = np.random.default_rng(seed)
    n_img = len(c["n_gt"])
    samples = {"ap50": [], "precision": [], "recall": [], "f1": [], "image_sensitivity": [], "count_mae": []}
    for _ in range(n):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
        samples["ap50"].append(average_precision(c, w))
        point = at_threshold(c, thr, w)
        for k in samples:
            if k != "ap50":
                samples[k].append(point[k])
    out = {}
    for k, vals in samples.items():
        vals = np.array(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [float("nan")] * 2
    return out


def recall_by_size(preds: dict, gts: dict, cls: int, iou_thr: float, thr: float, edges: list[float]) -> list[dict]:
    """Recall of ground-truth boxes grouped by their longer side in pixels."""
    sizes, hit = [], []
    for key, gt in gts.items():
        gt = gt[gt[:, 4] == cls] if len(gt) else np.zeros((0, 5))
        pred = preds.get(key, np.zeros((0, 6)))
        pred = pred[(pred[:, 5] == cls) & (pred[:, 4] >= thr)] if len(pred) else np.zeros((0, 6))
        _, matched = match_image(pred, gt, iou_thr)
        found = np.zeros(len(gt), dtype=bool)
        found[matched[matched >= 0]] = True
        sizes.extend(np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]))
        hit.extend(found)
    sizes, hit = np.array(sizes), np.array(hit, dtype=bool)
    rows = []
    bounds = [0.0, *edges, float("inf")]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        rows.append({"from_px": lo, "to_px": None if hi == float("inf") else hi, "boxes": int(mask.sum()),
                     "recall": float(hit[mask].mean()) if mask.any() else None})
    return rows
