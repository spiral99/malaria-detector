"""Make a small synthetic two-site dataset for the smoke test.

These are drawn blobs, not blood smears. They exist only to prove that every stage of the
pipeline runs end to end in a few minutes on a CPU. No result on them means anything.
The two "sites" differ in colour cast, blur and brightness, to mimic a change of clinic.
"""
from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from malaria import config

SITES = {
    "uganda": {"n": 72, "bg": (224, 206, 226), "parasite": (96, 40, 140), "wbc": (70, 50, 150), "blur": 0.6, "label": ("Trophozoite", "WBC")},
    "ghana": {"n": 48, "bg": (206, 214, 232), "parasite": (120, 60, 120), "wbc": (60, 70, 130), "blur": 1.3, "label": ("trophozoite", "wbc")},
}


def _voc(name: str, size: int, boxes) -> str:
    objects = "".join(
        f"<object><name>{label}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for label, x1, y1, x2, y2 in boxes
    )
    return f"<annotation><filename>{name}</filename><size><width>{size}</width><height>{size}</height><depth>3</depth></size>{objects}</annotation>"


def make_site(site: str, spec: dict, tmp: Path, size: int, rng: random.Random) -> list[Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(spec["n"]):
        jitter = lambda c: tuple(max(0, min(255, v + rng.randint(-10, 10))) for v in c)
        im = Image.new("RGB", (size, size), jitter(spec["bg"]))
        draw = ImageDraw.Draw(im)
        for _ in range(rng.randint(30, 60)):  # stain debris: pale specks that are not labelled
            x, y, r = rng.randint(0, size), rng.randint(0, size), rng.randint(1, 3)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=jitter((180, 160, 190)))
        boxes = []
        for _ in range(rng.randint(0, 7)):
            r = rng.randint(5, 9)
            x, y = rng.randint(r + 2, size - r - 2), rng.randint(r + 2, size - r - 2)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=jitter(spec["parasite"]))
            draw.ellipse([x - r // 3, y - r // 3, x + r // 3, y + r // 3], fill=jitter((230, 120, 170)))
            boxes.append((spec["label"][0], x - r - 1, y - r - 1, x + r + 1, y + r + 1))
        for _ in range(rng.randint(1, 4)):
            r = rng.randint(16, 24)
            x, y = rng.randint(r + 2, size - r - 2), rng.randint(r + 2, size - r - 2)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=jitter(spec["wbc"]))
            boxes.append((spec["label"][1], x - r - 1, y - r - 1, x + r + 1, y + r + 1))
        im = im.filter(ImageFilter.GaussianBlur(spec["blur"]))
        name = f"{site}_slide{i // 6:02d}_fov{i % 6}.jpg"
        im.save(tmp / name, quality=90)
        files.append(tmp / name)
        if not (site == "uganda" and i % 9 == 8):  # leave some images without annotations, like the real data
            (tmp / name).with_suffix(".xml").write_text(_voc(name, size, boxes))
            files.append((tmp / name).with_suffix(".xml"))
    return files


def run(cfg: config.Cfg, size: int = 320, seed: int = 7) -> None:
    raw = cfg.path("raw", make=True)
    rng = random.Random(seed)
    for site, spec in SITES.items():
        tmp = raw.parent / "_synthetic_tmp" / site
        files = make_site(site, spec, tmp, size, rng)
        archive = raw / f"Thick_{site.capitalize()}_synthetic.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for f in files:
                zf.write(f, f"{site}_thick/{f.name}")
        print(f"[synthetic] wrote {archive} ({spec['n']} images)")


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    run(config.from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
