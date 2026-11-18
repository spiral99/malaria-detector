import json
import zipfile
from collections import Counter

import pytest

from malaria import config
from malaria.data import annotations, download, prepare, synthetic, unpack

CLASSES = config.load()["classes"]


def test_label_map_handles_spellings_and_negatives():
    lm = annotations.LabelMap(CLASSES)
    assert lm.resolve("Trophozoite") == 0
    assert lm.resolve("white_blood_cell") == 1
    assert lm.resolve(" WBC ") == 1
    assert lm.resolve("NEG") == "neg"
    assert lm.resolve("gametocyte") is None


def test_parse_voc_csv_coco(tmp_path):
    (tmp_path / "a.xml").write_text(
        "<annotation><filename>A.JPG</filename><object><name>Trophozoite</name>"
        "<bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object></annotation>")
    (tmp_path / "boxes.csv").write_text("Image_ID,class,confidence,ymin,xmin,ymax,xmax\nb.jpg,WBC,1,5,6,50,60\nc.jpg,NEG,1,,,,\n")
    (tmp_path / "coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "d.png"}], "categories": [{"id": 7, "name": "trophozoite"}],
        "annotations": [{"image_id": 1, "category_id": 7, "bbox": [1, 2, 10, 20]}]}))
    (tmp_path / "notes.csv").write_text("slide,stain\n1,giemsa\n")
    (tmp_path / "classes.txt").write_text("Trophozoite\nWBC\n")
    (tmp_path / "e.txt").write_text("1 0.5 0.5 0.2 0.1\n0 0.1 0.1 0.05 0.05 0.9\n")
    (tmp_path / "readme.txt").write_text("notes about the data")
    store, counts = annotations.parse_tree(tmp_path)
    assert counts == {"voc": 1, "csv": 1, "coco": 1, "yolo": 1, "unrecognised": 1}
    e = store["e"].boxes
    assert e[0].label == "WBC" and e[0].normalized and abs(e[0].xmin - 0.4) < 1e-9 and abs(e[0].ymax - 0.55) < 1e-9
    assert e[1].label == "Trophozoite"
    assert (store["a"].boxes[0].xmin, store["a"].boxes[0].ymax) == (10, 40)
    assert (store["b"].boxes[0].xmin, store["b"].boxes[0].ymin, store["b"].boxes[0].xmax) == (6, 5, 60)
    assert store["c"].boxes[0].label == "NEG"
    assert (store["d"].boxes[0].xmax, store["d"].boxes[0].ymax) == (11, 22)


def test_clean_boxes_clips_drops_and_counts():
    lm = annotations.LabelMap(CLASSES)
    ann = annotations.ImageAnn("x", [
        annotations.Box("trophozoite", -5, 10, 20, 30),      # clipped at the left edge
        annotations.Box("wbc", 50, 50, 50.5, 80),             # too thin: dropped
        annotations.Box("schizont", 1, 1, 9, 9),              # unknown label
        annotations.Box("NEG", 0, 0, 0, 0),
    ])
    stats, unknown = Counter(), Counter()
    rows = prepare.clean_boxes(ann, lm, 100, 100, stats, unknown)
    assert rows == [(0, 0.0, 10.0, 20.0, 30.0)]
    norm = annotations.ImageAnn("y", [annotations.Box("1", 0.25, 0.5, 0.75, 1.0, normalized=True)])
    assert prepare.clean_boxes(norm, lm, 200, 100, Counter(), Counter()) == [(1, 50.0, 50.0, 150.0, 100.0)]
    assert stats["boxes_clipped"] == 1 and stats["boxes_dropped_degenerate"] == 1 and stats["negative_labels"] == 1
    assert unknown == {"schizont": 1}
    assert prepare.yolo_lines(rows, 100, 100) == ["0 0.100000 0.200000 0.200000 0.200000"]


def test_group_split_keeps_groups_together_and_is_deterministic():
    groups = {f"slide{i}": [f"slide{i}_{j}" for j in range(5)] for i in range(20)}
    fractions = {"train": 0.0, "val": 0.15, "test": 0.15}
    a = prepare.group_split(groups, fractions, seed=1)
    assert a == prepare.group_split(groups, fractions, seed=1)
    assert sum(len(v) for v in a.values()) == 100
    assert len(a["val"]) == 15 and len(a["test"]) == 15
    slide = lambda k: k.rsplit("_", 1)[0]
    assert not ({slide(k) for k in a["train"]} & {slide(k) for k in a["test"]})


def test_archive_helpers(tmp_path):
    assert unpack.site_of("Thick_Ghana.part1.rar", ["uganda", "ghana"]) == "ghana"
    assert unpack.is_first_volume("Thick_Ghana.part1.rar") and not unpack.is_first_volume("Thick_Ghana.part2.rar")
    assert unpack.is_first_volume("Thick_Uganda.rar")
    assert unpack.archive_stem("Thick_Ghana.part01.rar") == "Thick_Ghana"
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeError):
        unpack.extract(evil, tmp_path / "out")


def test_dataverse_listing_and_selection():
    payload = {"data": {"latestVersion": {"files": [
        {"dataFile": {"id": 1, "filename": "Thick_Ghana.part1.rar", "filesize": 10, "md5": "abc"}},
        {"dataFile": {"id": 2, "filename": "Thin_Uganda.rar", "filesize": 5, "checksum": {"type": "MD5", "value": "def"}}},
    ]}}}
    files = download.parse_listing(payload)
    assert [f["md5"] for f in files] == ["abc", "def"]
    assert [f["name"] for f in download.select(files, [], ["(?i)thin"])] == ["Thick_Ghana.part1.rar"]


def test_prepare_end_to_end_on_synthetic_data(tmp_path):
    cfg = config.load()
    cfg["paths"] = {k: str(tmp_path / k) for k in cfg["paths"]}
    cfg["split"]["adapt_pool"] = 12
    cfg["split"]["calib_pool"] = 8
    cfg["adapt"]["sizes"] = [6, 12]
    synthetic.run(cfg, size=96)
    unpack.run(cfg)
    report = prepare.run(cfg)
    uganda = report["sites"]["uganda"]["counts"]
    assert uganda["images_found"] == 72 and uganda["excluded_no_annotation"] == 8 and uganda["images_usable"] == 64
    splits = report["splits"]
    assert splits["uganda_train"] + splits["uganda_val"] + splits["uganda_test"] == 64
    assert splits["ghana_test"] + splits["ghana_adapt_pool"] + splits["ghana_calib"] == 48
    assert splits["ghana_adapt_pool"] == 12 and splits["ghana_calib"] == 8
    processed = tmp_path / "processed"
    lists = {name: set((processed / f"{name}.txt").read_text().split()) for name in splits}
    assert not (lists["ghana_test"] & lists["ghana_adapt_pool"])
    assert not (lists["ghana_calib"] & (lists["ghana_test"] | lists["ghana_adapt_pool"]))
    replay = lists["ghana_adapt_12_replay"]
    assert lists["ghana_adapt_12"] <= replay and len(replay) == 24
    assert replay - lists["ghana_adapt_12"] <= lists["uganda_train"]
    assert not (lists["uganda_train"] & (lists["uganda_val"] | lists["uganda_test"]))
    assert lists["ghana_adapt_6"] <= lists["ghana_adapt_12"] <= lists["ghana_adapt_pool"]
    first = sorted(lists["uganda_train"])[0]
    assert (processed / first[2:]).exists()
    assert (processed / first[2:].replace("/images/", "/labels/")).with_suffix(".txt").exists()


def test_shared_annotation_file_and_thin_images_are_handled(tmp_path):
    """One CSV for both sites in an archive that names no site; thin-smear folders are skipped."""
    from PIL import Image

    cfg = config.load()
    cfg["paths"] = {k: str(tmp_path / k) for k in cfg["paths"]}
    cfg["split"]["adapt_pool"] = 2
    cfg["adapt"]["sizes"] = [2]
    rows = ["Image_ID,class,xmin,ymin,xmax,ymax"]
    for site in ("Uganda", "Ghana"):
        for kind in ("Thick", "Thin"):
            folder = tmp_path / "interim" / "unknown" / "all" / site / kind
            folder.mkdir(parents=True)
            for i in range(8):
                name = f"{site[0]}{kind}{i}.jpg"
                im = Image.new("RGB", (64, 64), (200, 180, 220 if site == "Ghana" else 200))
                im.paste((90, 40, 140), (10, 10, 14 + 2 * i, 30))   # a different mark per image, so none are duplicates
                im.save(folder / name)
                rows.append(f"{name},Trophozoite,10,10,30,30")
    (tmp_path / "interim" / "unknown" / "labels.csv").write_text("\n".join(rows))
    report = prepare.run(cfg)
    assert report["sites"]["uganda"]["counts"]["images_found"] == 8
    assert report["sites"]["uganda"]["counts"]["images_usable"] == 8
    assert report["sites"]["ghana"]["counts"]["images_usable"] == 8
    assert report["sites"]["ghana"]["summary"]["boxes_trophozoite"] == 8


def test_exif_rotated_images_are_made_upright(tmp_path):
    """A photo stored landscape with a 'rotate 90' flag: labels use the rotated (portrait) frame."""
    from PIL import Image

    cfg = config.load()
    cfg["paths"] = {k: str(tmp_path / k) for k in cfg["paths"]}
    cfg["split"].update(adapt_pool=1, val_fraction=0.34, test_fraction=0.33)
    cfg["adapt"]["sizes"] = [1]
    for site in ("uganda", "ghana"):
        folder = tmp_path / "interim" / site / "x"
        folder.mkdir(parents=True)
        for i in range(3):
            im = Image.new("RGB", (120, 80), (200, 180, 220 if site == "ghana" else 200))
            im.paste((90, 40, 140), (100 + i, 5, 115 + i, 20))       # top-right corner of the raw frame
            exif = Image.Exif()
            exif[0x0112] = 6                                          # rotate 90 degrees clockwise when viewed
            im.save(folder / f"{site}{i}.jpg", exif=exif)
            # In the rotated 80x120 frame that mark sits in the bottom-right corner.
            (folder / f"{site}{i}.txt").write_text("0 0.84 0.90 0.18 0.12\n")
    report = prepare.run(cfg)
    assert report["sites"]["ghana"]["counts"]["images_rotated_upright"] == 3
    with Image.open(tmp_path / "processed" / "ghana" / "images" / "ghana0.jpg") as im:
        assert im.size == (80, 120) and im.getexif().get(0x0112) in (None, 1)
        assert im.getpixel((70, 110))[2] > 100 and im.getpixel((70, 110))[0] < 130   # the mark is where the label says
    row = next(r for r in __import__("csv").DictReader(open(tmp_path / "processed" / "index.csv")) if r["key"] == "ghana0")
    assert (row["width"], row["height"]) == ("80", "120")
