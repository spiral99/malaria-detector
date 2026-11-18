"""Run the exported ONNX detector on CPU with ONNX Runtime. No PyTorch needed at serving time."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

PAD = 114


def letterbox(img: Image.Image, size: int):
    """Resize keeping the aspect ratio and pad to a square. Returns (CHW float array, scale, pad_x, pad_y)."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (size, size), (PAD, PAD, PAD))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return arr, scale, pad_x, pad_y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        iou = inter / np.maximum(area(boxes[i]) + area(boxes[rest]) - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode(output: np.ndarray, n_classes: int, min_conf: float, iou_thr: float) -> np.ndarray:
    """Turn the raw model output into rows [x1, y1, x2, y2, confidence, class] in letterboxed pixels."""
    out = output[0]
    if out.ndim == 2 and out.shape[-1] == 6 and out.shape[0] != 4 + n_classes:
        rows = out[out[:, 4] >= min_conf]  # end-to-end head: already x1, y1, x2, y2, conf, class
        return rows.astype(np.float32)
    if out.shape[0] == 4 + n_classes:
        out = out.T  # (anchors, 4 + classes)
    scores = out[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    mask = conf >= min_conf
    out, cls, conf = out[mask], cls[mask], conf[mask]
    boxes = np.stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2,
                      out[:, 0] + out[:, 2] / 2, out[:, 1] + out[:, 3] / 2], axis=1)
    rows = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for k in nms(boxes[idx], conf[idx], iou_thr):
            rows.append([*boxes[idx[k]], conf[idx[k]], c])
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


class Detector:
    def __init__(self, model_path: str | Path, meta_path: str | Path):
        import onnxruntime as ort

        self.meta = json.loads(Path(meta_path).read_text())
        self.classes = self.meta["classes"]
        self.imgsz = int(self.meta["imgsz"])
        self.threshold = float(self.meta["conf_threshold"])
        self.nms_iou = float(self.meta.get("nms_iou", 0.6))
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img: Image.Image, conf: float | None = None) -> np.ndarray:
        """Return rows [x1, y1, x2, y2, confidence, class] in the original image's pixels."""
        thr = self.threshold if conf is None else conf
        arr, scale, pad_x, pad_y = letterbox(img, self.imgsz)
        output = self.session.run(None, {self.input_name: arr[None]})[0]
        rows = decode(output, len(self.classes), thr, self.nms_iou)
        if len(rows):
            rows[:, [0, 2]] = np.clip((rows[:, [0, 2]] - pad_x) / scale, 0, img.size[0])
            rows[:, [1, 3]] = np.clip((rows[:, [1, 3]] - pad_y) / scale, 0, img.size[1])
        return rows
