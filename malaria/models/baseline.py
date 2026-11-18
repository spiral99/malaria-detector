"""Baseline: a small CNN that classifies image patches as parasite or not.

It sets a floor and answers a narrower question than the detector: given a crop, is there a
parasite in it? Negatives are random background crops plus every white-blood-cell crop (the
hardest look-alike). It is trained on the training site and scored on both sites. Its scores
are patch-level and are not comparable with the detector's AP; the comparison of interest is
whether it loses accuracy across sites in the same way.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from PIL import Image

from malaria import config, evalcore, tracking
from malaria.models.evaluate import load_gt, split_items


def sample_patches(cfg: config.Cfg, split: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    b = cfg["baseline"]
    size, ctx = b["patch"], b["context"]
    items = split_items(cfg, split)
    gts = load_gt(items)
    xs, ys = [], []
    all_sides = [max(g[2] - g[0], g[3] - g[1]) for gt in gts.values() for g in gt if g[4] == 0]
    default_side = float(np.median(all_sides)) if all_sides else 32.0
    order = list(items)
    rng.shuffle(order)
    for it in order:
        if len(xs) >= b["max_patches_per_split"]:
            break
        gt = gts[it["key"]]
        parasites, wbcs = gt[gt[:, 4] == 0], gt[gt[:, 4] == 1]
        img = Image.open(it["image"]).convert("RGB")
        w, h = img.size

        def crop(cx, cy, side):
            half = side * ctx / 2
            return np.asarray(img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((size, size), Image.BILINEAR))

        for g in parasites:
            xs.append(crop((g[0] + g[2]) / 2, (g[1] + g[3]) / 2, max(g[2] - g[0], g[3] - g[1])))
            ys.append(1)
        for g in wbcs[:3]:
            xs.append(crop((g[0] + g[2]) / 2, (g[1] + g[3]) / 2, default_side))
            ys.append(0)
        wanted, tries = max(2, b["neg_per_pos"] * len(parasites)), 0
        while wanted and tries < 50:
            tries += 1
            side = default_side * rng.uniform(0.8, 1.25)
            cx, cy = rng.uniform(side, w - side), rng.uniform(side, h - side)
            box = np.array([[cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]])
            if len(parasites) and evalcore.iou_matrix(box, parasites[:, :4]).max() > 0:
                continue
            xs.append(crop(cx, cy, side))
            ys.append(0)
            wanted -= 1
    return np.stack(xs).astype(np.uint8), np.array(ys, dtype=np.int64)


def build_net():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1))


def run(cfg: config.Cfg) -> dict:
    import torch
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    s, b = cfg["split"], cfg["baseline"]
    torch.manual_seed(s["seed"])
    rng = random.Random(s["seed"])
    names = {"train": f"{s['train_site']}_train", "val": f"{s['train_site']}_val",
             "in_site_test": f"{s['train_site']}_test", "cross_site_test": f"{s['test_site']}_test"}
    data = {k: sample_patches(cfg, v, rng) for k, v in names.items()}
    device = "cuda" if torch.cuda.is_available() and cfg["train"].get("device") != "cpu" else "cpu"

    def tensor(x):
        return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255)

    net = build_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=b["lr"])
    x_train, y_train = tensor(data["train"][0]), torch.from_numpy(data["train"][1]).float()
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def scores(x):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                out.append(torch.sigmoid(net(tensor(x[i:i + 512]).to(device))).squeeze(1).cpu().numpy())
        return np.concatenate(out)

    best_auc, best_state = -1.0, None
    for epoch in range(b["epochs"]):
        net.train()
        perm = torch.randperm(len(x_train))
        for i in range(0, len(perm), b["batch"]):
            idx = perm[i:i + b["batch"]]
            xb = x_train[idx]
            if torch.rand(1).item() < 0.5:
                xb = xb.flip(3)
            opt.zero_grad()
            loss = loss_fn(net(xb.to(device)).squeeze(1), y_train[idx].to(device))
            loss.backward()
            opt.step()
        auc = roc_auc_score(data["val"][1], scores(data["val"][0])) if len(set(data["val"][1])) > 1 else 0.0
        print(f"[baseline] epoch {epoch + 1}/{b['epochs']}  val AUROC {auc:.3f}")
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    val_scores = scores(data["val"][0])
    grid = np.linspace(0.05, 0.95, 19)
    thr = float(grid[int(np.argmax([f1_score(data["val"][1], val_scores >= t, zero_division=0) for t in grid]))])
    result = {"threshold": thr, "note": "patch-level scores; not comparable with detection AP", "splits": {}}
    for key in ("val", "in_site_test", "cross_site_test"):
        x, y = data[key]
        p = scores(x)
        two = len(set(y)) > 1
        result["splits"][key] = {
            "patches": int(len(y)), "positives": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)) if two else None,
            "average_precision": float(average_precision_score(y, p)) if two else None,
            "f1": float(f1_score(y, p >= thr, zero_division=0)),
        }
    torch.save(net.state_dict(), cfg.path("models", "baseline_cnn.pt", make=True))
    out = cfg.path("reports", "metrics", "baseline.json", make=True)
    out.write_text(json.dumps(result, indent=2))
    with tracking.run("baseline-patch-cnn", dict(b)) as tracker:
        for key, vals in result["splits"].items():
            tracker.metrics(vals, prefix=f"{key}_")
    print(f"[baseline] in-site AUROC {result['splits']['in_site_test']['auroc']}, cross-site AUROC {result['splits']['cross_site_test']['auroc']}")
    return result


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    run(config.from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
