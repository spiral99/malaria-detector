import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from service import app as service_app
from service import inference, quality


class FakeDetector:
    classes = ["trophozoite", "wbc"]
    threshold = 0.4
    meta = {"name": "fake", "trained_on": "uganda", "imgsz": 64, "low_conf_margin": 0.15,
            "quality_ranges": {"brightness": [0.2, 0.9]}}

    def predict(self, img, conf=None):
        rows = np.array([[10, 10, 20, 20, 0.95, 0], [30, 30, 40, 40, 0.45, 0], [50, 50, 90, 90, 0.9, 1]], dtype=np.float32)
        return rows[rows[:, 4] >= (self.threshold if conf is None else conf)]


def _png(colour=(200, 180, 210)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (120, 100), colour).save(buf, "PNG")
    return buf.getvalue()


def _client(detector):
    client = TestClient(service_app.app)
    client.__enter__()
    service_app.state.update(detector=detector, error=None if detector else "FileNotFoundError: no model")
    return client


def test_predict_returns_counts_boxes_and_flags():
    client = _client(FakeDetector())
    body = client.post("/predict", files={"file": ("smear.png", _png(), "image/png")}).json()
    assert body["counts"] == {"trophozoite": 2, "wbc": 1}
    assert body["parasites_per_wbc"] == 2.0
    assert body["image"] == {"width": 120, "height": 100}
    assert body["detections"][0]["confidence"] == 0.95 and len(body["detections"][0]["box"]) == 4
    assert body["low_confidence"] and "close to the decision threshold" in body["low_confidence_reasons"][0]


def test_conf_parameter_and_dark_image_flag():
    client = _client(FakeDetector())
    body = client.post("/predict?conf=0.8", files={"file": ("smear.png", _png((5, 5, 5)), "image/png")}).json()
    assert body["counts"]["trophozoite"] == 1 and body["threshold"] == 0.8
    assert any("unlike the training images" in r for r in body["low_confidence_reasons"])
    assert client.post("/predict?conf=1.5", files={"file": ("s.png", _png(), "image/png")}).status_code == 422


def test_bad_uploads_and_missing_model():
    client = _client(FakeDetector())
    assert client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 415
    assert client.post("/predict", files={"file": ("x.png", b"", "image/png")}).status_code == 400
    annotated = client.post("/predict/annotated", files={"file": ("smear.png", _png(), "image/png")})
    assert annotated.status_code == 200 and annotated.headers["content-type"] == "image/png"
    assert client.get("/health").json()["status"] == "ok"
    assert "Malaria" in client.get("/").text
    client = _client(None)
    assert client.get("/health").json()["status"] == "no_model"
    assert client.post("/predict", files={"file": ("smear.png", _png(), "image/png")}).status_code == 503


def test_letterbox_and_decode_round_trip():
    arr, scale, pad_x, pad_y = inference.letterbox(Image.new("RGB", (200, 100)), 64)
    assert arr.shape == (3, 64, 64) and abs(scale - 0.32) < 1e-9 and (pad_x, pad_y) == (0, 16)
    raw = np.zeros((1, 6, 4), dtype=np.float32)          # (batch, 4 + classes, anchors)
    raw[0, :, 0] = [20, 20, 10, 10, 0.9, 0.0]             # parasite
    raw[0, :, 1] = [21, 20, 10, 10, 0.8, 0.0]             # duplicate of it: removed by NMS
    raw[0, :, 2] = [50, 50, 20, 20, 0.1, 0.7]             # white cell
    raw[0, :, 3] = [40, 10, 6, 6, 0.05, 0.02]             # below the threshold
    rows = inference.decode(raw, 2, 0.25, 0.6)
    assert sorted(rows[:, 5].tolist()) == [0.0, 1.0]
    assert np.allclose(rows[rows[:, 5] == 0][0, :5], [15, 15, 25, 25, 0.9])
    end_to_end = np.array([[[1, 2, 3, 4, 0.9, 1], [0, 0, 0, 0, 0.0, 0]]], dtype=np.float32)
    assert inference.decode(end_to_end, 2, 0.25, 0.6).tolist() == [[1, 2, 3, 4, 0.8999999761581421, 1]]


def test_quality_features_separate_sharp_from_flat():
    flat = Image.new("RGB", (64, 64), (128, 128, 128))
    noisy = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    assert quality.features(noisy)["sharpness"] > quality.features(flat)["sharpness"]
    assert quality.outside_range({"brightness": 0.05}, {"brightness": [0.2, 0.9]}) == ["brightness"]
