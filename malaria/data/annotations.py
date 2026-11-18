"""Read bounding-box annotations in the formats the dataset may ship in.

Supported: Pascal VOC XML (one file per image, as written by LabelImg), CSV tables with one box
per row (the Zindi Train.csv layout included), COCO JSON, and YOLO text files (one per image, with
normalised coordinates; class names are read from a classes.txt / obj.names file if one exists,
otherwise class indices are used in the order given in params.yaml). Everything is converted to one in-memory form: image key -> list of boxes
in pixel coordinates. The image key is the lower-cased file name without its extension.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_COLS = ("image_id", "imageid", "image", "filename", "file_name", "file", "image_name", "imagename", "image_path", "path")
LABEL_COLS = ("class", "label", "name", "category", "class_name", "classname")


CLASS_LIST_FILES = ("classes.txt", "obj.names", "labels.txt", "class_names.txt", "names.txt")


@dataclass
class Box:
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    normalized: bool = False   # True when coordinates are fractions of the image size (YOLO files)


@dataclass
class ImageAnn:
    key: str
    boxes: list[Box] = field(default_factory=list)
    negative: bool = False          # an explicit "nothing in this image" label was seen
    sources: set = field(default_factory=set)


def image_key(name: str) -> str:
    return Path(str(name).replace("\\", "/")).stem.strip().lower()


class LabelMap:
    """Maps raw label spellings to class indices using the aliases in params.yaml."""

    def __init__(self, classes_cfg: dict):
        self.names = list(classes_cfg["names"])
        self.lookup = {}
        for idx, name in enumerate(self.names):
            for alias in [name, *classes_cfg.get("aliases", {}).get(name, [])]:
                self.lookup[self._norm(alias)] = idx
        self.negative = {self._norm(x) for x in classes_cfg.get("negative_labels", [])}

    @staticmethod
    def _norm(label: str) -> str:
        return " ".join(str(label).replace("_", " ").replace("-", " ").lower().split())

    def resolve(self, label: str):
        """Return a class index, the string 'neg', or None for a label we do not know."""
        norm = self._norm(label)
        if norm in self.lookup:
            return self.lookup[norm]
        if norm in self.negative:
            return "neg"
        if norm.isdigit() and int(norm) < len(self.names):   # a bare YOLO class index
            return int(norm)
        return None


def _get(store: dict, key: str, source: str) -> ImageAnn:
    ann = store.setdefault(key, ImageAnn(key))
    ann.sources.add(source)
    return ann


def parse_voc(path: Path, store: dict) -> bool:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return False
    if root.tag != "annotation":
        return False
    ann = _get(store, image_key(path.name), "voc")
    filename = root.findtext("filename")
    if filename and image_key(filename) != ann.key:
        store.setdefault("__alias__", {})[image_key(filename)] = ann.key
    for obj in root.iter("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            coords = [float(box.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        ann.boxes.append(Box((obj.findtext("name") or "").strip(), *coords))
    return True


def parse_csv(path: Path, store: dict) -> bool:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return False
        cols = {c.strip().lower(): c for c in reader.fieldnames if c}
        image_col = next((cols[c] for c in IMAGE_COLS if c in cols), None)
        label_col = next((cols[c] for c in LABEL_COLS if c in cols), None)
        corners = all(c in cols for c in ("xmin", "ymin", "xmax", "ymax"))
        xywh = all(c in cols for c in ("x", "y")) and (("w" in cols and "h" in cols) or ("width" in cols and "height" in cols))
        if image_col is None or label_col is None or not (corners or xywh):
            return False
        for row in reader:
            if not row.get(image_col):
                continue
            ann = _get(store, image_key(row[image_col]), "csv")
            label = (row.get(label_col) or "").strip()
            try:
                if corners:
                    coords = [float(row[cols[c]]) for c in ("xmin", "ymin", "xmax", "ymax")]
                else:
                    x, y = float(row[cols["x"]]), float(row[cols["y"]])
                    w = float(row[cols["w" if "w" in cols else "width"]])
                    h = float(row[cols["h" if "h" in cols else "height"]])
                    coords = [x, y, x + w, y + h]
            except (TypeError, ValueError):
                coords = None
            if coords is None or any(c != c for c in coords):  # missing or NaN: image-level label only
                ann.boxes.append(Box(label, 0.0, 0.0, 0.0, 0.0))
            else:
                ann.boxes.append(Box(label, *coords))
    return True


def parse_coco(path: Path, store: dict) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict) or not all(k in data for k in ("images", "annotations", "categories")):
        return False
    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {im["id"]: image_key(im["file_name"]) for im in data["images"]}
    for key in images.values():
        _get(store, key, "coco")
    for a in data["annotations"]:
        if a.get("image_id") not in images or "bbox" not in a:
            continue
        x, y, w, h = a["bbox"]
        store[images[a["image_id"]]].boxes.append(Box(cats.get(a.get("category_id"), ""), x, y, x + w, y + h))
    return True


def parse_yolo(path: Path, store: dict, class_names: list[str] | None) -> bool:
    """One YOLO label file: lines of `class cx cy w h [conf]` with coordinates in 0-1."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    rows = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) not in (5, 6) or not parts[0].lstrip("-").isdigit():
            return False
        try:
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            return False
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            return False
        idx = int(parts[0])
        label = class_names[idx] if class_names and 0 <= idx < len(class_names) else str(idx)
        rows.append(Box(label, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, normalized=True))
    ann = _get(store, image_key(path.name), "yolo")
    ann.boxes.extend(rows)
    return True


def find_class_list(root: Path) -> list[str] | None:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.lower() in CLASS_LIST_FILES:
            names = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            if names:
                return names
    return None


def parse_tree(root: Path) -> tuple[dict, dict]:
    """Scan a folder for annotation files. Returns (annotations by image key, counts per format)."""
    store: dict = {}
    counts = {"voc": 0, "csv": 0, "coco": 0, "yolo": 0, "unrecognised": 0}
    class_names = find_class_list(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".xml":
            counts["voc" if parse_voc(path, store) else "unrecognised"] += 1
        elif suffix == ".csv":
            counts["csv" if parse_csv(path, store) else "unrecognised"] += 1
        elif suffix == ".json":
            counts["coco" if parse_coco(path, store) else "unrecognised"] += 1
        elif suffix == ".txt" and path.name.lower() not in CLASS_LIST_FILES and "readme" not in path.name.lower():
            counts["yolo" if parse_yolo(path, store, class_names) else "unrecognised"] += 1
    aliases = store.pop("__alias__", {})
    for alias, key in aliases.items():
        store.setdefault(alias, store[key])
    return store, counts
