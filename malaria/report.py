"""Final stage: turn the metric files into figures and result tables.

Writes reports/results.md and reports/figures/*.png, and refreshes the results block between
the RESULTS markers in docs/REPORT.md. Text outside the markers is never touched.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from malaria import config

IN_SITE, CROSS = "#2a78d6", "#eb6834"      # fixed colour per site, used in every figure
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _load(cfg: config.Cfg, name: str):
    path = cfg.path("reports", "metrics", name)
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None or value != value else f"{value:.{digits}f}"


def _ci(pair) -> str:
    return f"{_fmt(pair[0], 2)} to {_fmt(pair[1], 2)}"


def _axes(title: str, ylabel: str, figsize=(7.2, 4.2)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax, plt


def _save(fig, plt, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def _paired_bars(ax, labels, in_vals, cross_vals, in_err=None, cross_err=None, names=("In-site", "Cross-site")):
    import numpy as np

    x = np.arange(len(labels))
    width = 0.36
    for offset, vals, err, colour, name in ((-width / 2 - 0.01, in_vals, in_err, IN_SITE, names[0]),
                                            (width / 2 + 0.01, cross_vals, cross_err, CROSS, names[1])):
        clean = [0 if v is None or v != v else v for v in vals]
        bars = ax.bar(x + offset, clean, width, color=colour, label=name)
        if err:
            lo = [max(0, v - e[0]) for v, e in zip(clean, err)]
            hi = [max(0, e[1] - v) for v, e in zip(clean, err)]
            ax.errorbar(x + offset, clean, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        for rect, v in zip(bars, vals):
            ax.annotate(_fmt(v, 2), (rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, 12 if err else 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.1))


def _adapted(det: dict) -> dict:
    """{mode: [(n, model name), ...]} for the adapted models, sorted by n."""
    out: dict = {}
    for m in det["models"]:
        if not m.startswith("adapt_"):
            continue
        parts = m.split("_")
        out.setdefault(parts[2] if len(parts) > 2 else "plain", []).append((int(parts[1]), m))
    return {k: sorted(v) for k, v in out.items()}


def _cross(entry: dict, det: dict, recalibrated: bool) -> dict:
    if recalibrated and entry.get("cross_site_recalibrated"):
        return entry["cross_site_recalibrated"]
    return entry["splits"][det["splits"]["cross_site_test"]]


def figures(cfg: config.Cfg, det, shift) -> dict:
    out = {}
    fig_dir = cfg.path("reports", "figures", make=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    s = cfg["split"]
    a, b = s["train_site"].capitalize(), s["test_site"].capitalize()
    names = (f"In-site ({a})", f"Cross-site ({b})")
    if det:
        parasite = cfg.class_names[0]
        in_split, cross_split = det["splits"]["in_site_test"], det["splits"]["cross_site_test"]
        row = lambda m, split: det["models"][m]["splits"][split]["classes"][parasite]
        plain = [m for m, e in det["models"].items() if e.get("kind", "trained") in ("trained", "resized")]
        fig, ax, plt = _axes("Parasite detection: same clinic versus a new clinic", "AP50 (higher is better)")
        _paired_bars(ax, plain, [row(m, in_split)["ap50"] for m in plain], [row(m, cross_split)["ap50"] for m in plain],
                     [row(m, in_split)["ci95"]["ap50"] for m in plain], [row(m, cross_split)["ci95"]["ap50"] for m in plain], names)
        ax.set_ylim(0, 1.1)
        out["in_vs_cross"] = _save(fig, plt, fig_dir / "in_vs_cross.png")

        sel = det["selected"]
        adapted = _adapted(det)
        if adapted:
            fig, ax, plt = _axes(f"Fine-tuning on labelled {b} images: gain there, cost at home", "Parasite AP50")
            styles = {"plain": ("-", "o", "plain"), "replay": ("--", "s", f"with {a} replay")}
            for mode, models in adapted.items():
                line, marker, label = styles.get(mode, (":", "^", mode))
                xs = [0] + [n for n, _ in models]
                ms = [sel] + [m for _, m in models]
                for split, colour, site in ((cross_split, CROSS, b), (in_split, IN_SITE, a)):
                    ys = [row(m, split)["ap50"] for m in ms]
                    ax.plot(xs, ys, color=colour, linestyle=line, linewidth=2, marker=marker, markersize=7,
                            markeredgecolor=SURFACE, markeredgewidth=2, label=f"{site} test, {label}")
                    ax.annotate(_fmt(ys[-1], 2), (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                                va="center", fontsize=8, color=INK)
            ax.set_xticks([0] + [n for n, _ in next(iter(adapted.values()))])
            ax.set_xlabel(f"Labelled {b} images used for fine-tuning", color=MUTED, fontsize=9)
            ax.set_ylim(0, 1.1)
            top = max(n for v in adapted.values() for n, _ in v)
            ax.set_xlim(-0.04 * top, top * 1.12)
            ax.legend(frameon=False, fontsize=8, loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.16))
            out["adaptation"] = _save(fig, plt, fig_dir / "adaptation.png")

        sel_entry = det["models"][sel]
        bins_in = sel_entry["splits"][in_split]["recall_by_size"]
        bins_cross = _cross(sel_entry, det, True)["recall_by_size"]
        labels = [f"{_fmt(r['from_px'], 0)}–{_fmt(r['to_px'], 0)} px" if r["to_px"] else f"over {_fmt(r['from_px'], 0)} px" for r in bins_in]
        labels[0] = f"under {_fmt(bins_in[0]['to_px'], 0)} px"
        fig, ax, plt = _axes(f"Parasite recall by box size ({sel})", "Recall at each site's own threshold")
        _paired_bars(ax, labels, [r["recall"] for r in bins_in], [r["recall"] for r in bins_cross], names=names)
        ax.set_ylim(0, 1.1)
        out["recall_by_size"] = _save(fig, plt, fig_dir / "recall_by_size.png")
    if shift:
        feats = shift["features"]
        fig, ax, plt = _axes("How different the photos look, before any labels", "KS statistic (0 = same, 1 = fully separate)")
        _paired_bars(ax, [f.replace("_", " ") for f in feats],
                     [shift["control_train_vs_in_site_test"]["ks"][f] for f in feats],
                     [shift["train_vs_cross_site_test"]["ks"][f] for f in feats],
                     names=(f"{a} train vs {a} test (control)", f"{a} train vs {b} test"))
        ax.set_ylim(0, 1.1)
        out["shift"] = _save(fig, plt, fig_dir / "shift.png")
    return out


def results_markdown(cfg: config.Cfg, figs: dict, prefix: str = "") -> str:
    det, base, shift = _load(cfg, "detection.json"), _load(cfg, "baseline.json"), _load(cfg, "shift.json")
    data_path = cfg.path("reports", "data_report.json")
    data = json.loads(data_path.read_text()) if data_path.exists() else None
    s = cfg["split"]
    lines = []
    if data:
        lines += ["### Data after cleaning", "", "| Site | Images found | No annotation | Duplicates | Unreadable | Usable | Parasite boxes | WBC boxes |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for site, info in data["sites"].items():
            c, sm = info["counts"], info.get("summary", {})
            lines.append(f"| {site} | {c.get('images_found', 0)} | {c.get('excluded_no_annotation', 0)} | "
                         f"{c.get('excluded_exact_duplicate', 0) + c.get('excluded_duplicate_name', 0)} | {c.get('excluded_unreadable', 0)} | "
                         f"{c.get('images_usable', 0)} | {sm.get('boxes_' + cfg.class_names[0], 0)} | {sm.get('boxes_' + cfg.class_names[1], 0)} |")
        lines += ["", "Split sizes: " + ", ".join(f"{k} {v}" for k, v in data["splits"].items()) + ".", ""]
    if det:
        parasite, wbc = cfg.class_names[0], cfg.class_names[1]
        in_split, cross_split = det["splits"]["in_site_test"], det["splits"]["cross_site_test"]
        calib = det["splits"].get("cross_site_calib")
        sel = det["selected"]
        a_site, b_site = s["train_site"], s["test_site"]

        def cells(r):
            return (f"{_fmt(r['ap50'])} ({_ci(r['ci95']['ap50'])}) | {_fmt(r['ap50_95'])} | {_fmt(r['precision'])} | "
                    f"{_fmt(r['recall'])} | {_fmt(r['f1'])} | {_fmt(r['image_sensitivity'])} | {_fmt(r['count_mae'], 2)}")

        lines += ["### Parasite detection, in-site and cross-site", "",
                  f"A detection counts when it overlaps a labelled box by IoU ≥ {det['iou']}. Intervals are 95% bootstrap "
                  f"intervals over images. The model selected on `{det['splits']['val']}` is **{sel}**.", "",
                  f"- **As shipped**: threshold chosen on `{det['splits']['val']}`, what a new clinic gets with no labels."]
        if calib:
            lines.append(f"- **Recalibrated**: threshold chosen on `{calib}`, labelled {b_site} images that are never trained on "
                         f"and never in the test set. AP does not depend on the threshold; the other columns do.")
        lines += ["- `name@size` is the same trained model run at a larger input size, with no retraining.", "",
                  "| Model | Test set | Threshold | AP50 (95% interval) | AP50-95 | Precision | Recall | F1 | Image sensitivity | Count error (MAE) |",
                  "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for model, entry in det["models"].items():
            if entry.get("kind") == "adapted":
                continue
            lines.append(f"| {model} | {a_site} | {_fmt(entry['threshold'], 2)} | {cells(entry['splits'][in_split]['classes'][parasite])} |")
            lines.append(f"| {model} | {b_site}, as shipped | {_fmt(entry['threshold'], 2)} | {cells(entry['splits'][cross_split]['classes'][parasite])} |")
            if entry.get("cross_site_recalibrated"):
                lines.append(f"| {model} | {b_site}, recalibrated | {_fmt(entry['threshold_local'], 2)} | "
                             f"{cells(entry['cross_site_recalibrated']['classes'][parasite])} |")
        ia = det["models"][sel]["splits"][in_split]["classes"][parasite]
        ib = det["models"][sel]["splits"][cross_split]["classes"][parasite]
        if ia["ap50"] == ia["ap50"] and ib["ap50"] == ib["ap50"] and ia["ap50"] > 0:
            lines += ["", f"For {sel}, AP50 goes from {_fmt(ia['ap50'])} in-site to {_fmt(ib['ap50'])} cross-site, "
                          f"a relative change of {100 * (ib['ap50'] - ia['ap50']) / ia['ap50']:+.1f}%."]
        if "in_vs_cross" in figs:
            lines += ["", f"![In-site versus cross-site AP50]({prefix}figures/{figs['in_vs_cross']})"]

        sel_e = det["models"][sel]
        lines += ["", "### White blood cells and error analysis", "",
                  f"For {sel}. {b_site.capitalize()} rows use the recalibrated threshold where one exists.", "",
                  "| Test set | WBC AP50 | WBC recall | Parasite false positives | of which on a labelled WBC |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for label, block in ((a_site, sel_e["splits"][in_split]), (b_site, _cross(sel_e, det, True))):
            lines.append(f"| {label} | {_fmt(block['classes'][wbc]['ap50'])} | {_fmt(block['classes'][wbc]['recall'])} | "
                         f"{block['wbc_confusion']['parasite_false_positives']} | {block['wbc_confusion']['on_a_labelled_wbc']} |")
        if "recall_by_size" in figs:
            lines += ["", f"![Recall by parasite box size]({prefix}figures/{figs['recall_by_size']})"]
        worst = det.get("worst_images", {})
        if any(worst.values()):
            lines += ["", f"The images with the most errors for {sel} are saved in `reports/figures/errors/` "
                          f"({', '.join(f'{len(v)} for {k}' for k, v in worst.items())}). "
                          "Green = correct detection, red = false alarm, yellow = missed parasite."]

        adapted = _adapted(det)
        if adapted:
            lines += ["", "### Few-shot adaptation to the new site", "",
                      f"{b_site.capitalize()} columns use each model's recalibrated threshold; {a_site.capitalize()} columns use its home threshold. "
                      f"Replay mixes in the same number of {a_site.capitalize()} training images.", "",
                      f"| Labelled {b_site} images | Mode | {b_site} AP50 | {b_site} precision | {b_site} recall | "
                      f"{b_site} image sensitivity | {b_site} count error | {a_site} AP50 | {a_site} recall |",
                      "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            rows = [(0, "none", sel)] + [(n, mode, m) for mode, models in adapted.items() for n, m in models]
            for n, mode, m in rows:
                e_ = det["models"][m]
                c = _cross(e_, det, True)["classes"][parasite]
                h = e_["splits"][in_split]["classes"][parasite]
                lines.append(f"| {n} | {mode} | {_fmt(c['ap50'])} | {_fmt(c['precision'])} | {_fmt(c['recall'])} | "
                             f"{_fmt(c['image_sensitivity'])} | {_fmt(c['count_mae'], 1)} | {_fmt(h['ap50'])} | {_fmt(h['recall'])} |")
            if "adaptation" in figs:
                lines += ["", f"![Adaptation curve]({prefix}figures/{figs['adaptation']})"]
    if base:
        lines += ["", "### Baseline: patch classifier", "", "Patch-level scores for a small CNN. They are not comparable with detection AP.", "",
                  "| Test set | Patches | AUROC | Average precision | F1 |", "| --- | ---: | ---: | ---: | ---: |"]
        for key, label in (("in_site_test", "in-site"), ("cross_site_test", "cross-site")):
            r = base["splits"][key]
            lines.append(f"| {label} | {r['patches']} | {_fmt(r['auroc'])} | {_fmt(r['average_precision'])} | {_fmt(r['f1'])} |")
    if shift:
        ctrl, cross = shift["control_train_vs_in_site_test"], shift["train_vs_cross_site_test"]
        lines += ["", "### Site shift measured without labels", "",
                  f"A classifier given only simple image statistics separates the two sites with AUROC {_fmt(cross['site_classifier_auroc'])} "
                  f"(control between two splits of the training site: {_fmt(ctrl['site_classifier_auroc'])}; 0.5 means it cannot tell them apart). "
                  f"{100 * shift['cross_site_images_outside_training_range']:.0f}% of cross-site test images fall outside the training range on at least one statistic "
                  f"(in-site: {100 * shift['in_site_images_outside_training_range']:.0f}%)."]
        if "shift" in figs:
            lines += ["", f"![Site shift by image statistic]({prefix}figures/{figs['shift']})"]
    return "\n".join(lines) + "\n"


def run(cfg: config.Cfg) -> Path:
    figs = figures(cfg, _load(cfg, "detection.json"), _load(cfg, "shift.json"))
    out = cfg.path("reports", "results.md", make=True)
    out.write_text(results_markdown(cfg, figs), encoding="utf-8")
    print(f"[report] wrote {out} and {len(figs)} figure(s)")
    final = config.ROOT / "docs" / "REPORT.md"
    if cfg.path("reports") == config.ROOT / "reports" and final.exists():
        text = final.read_text(encoding="utf-8")
        if START in text and END in text:
            block = f"{START}\n{results_markdown(cfg, figs, prefix='../reports/')}{END}"
            final.write_text(re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S), encoding="utf-8")
            print(f"[report] refreshed the results block in {final}")
    return out


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    run(config.from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
