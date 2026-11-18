"""FastAPI service: send a thick blood-smear photo, get parasite and white-cell detections back.

Run locally:   uvicorn service.app:app --port 8000
Run in Docker: docker compose up --build
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from service import quality
from service.inference import Detector

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "model.onnx"))
META_PATH = Path(os.environ.get("META_PATH", ROOT / "models" / "model_meta.json"))
MAX_BYTES = 25 * 1024 * 1024
COLOURS = {0: (235, 104, 52), 1: (42, 120, 214)}
state: dict = {"detector": None, "error": None}


def load_model() -> None:
    try:
        state["detector"] = Detector(MODEL_PATH, META_PATH)
        state["error"] = None
    except Exception as exc:  # the API still starts, and /health explains what is missing
        state["detector"] = None
        state["error"] = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    url = os.environ.get("PUBLIC_URL")
    if url:
        status = "model loaded" if state["detector"] else f"no model ({state['error']})"
        print(f"Malaria detector ready ({status}). Open {url} in your browser; API docs at {url}/docs", flush=True)
    yield


app = FastAPI(
    title="Malaria parasite detector",
    description="Detects malaria parasites (trophozoites) and white blood cells in thick blood-smear photos. "
                "A decision-support aid for trained staff. It is not a diagnostic device.",
    version="1.0.0",
    lifespan=lifespan,
)


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]  # x1, y1, x2, y2 in pixels of the uploaded image


class Prediction(BaseModel):
    image: dict
    detections: list[Detection]
    counts: dict
    parasites_per_wbc: float | None
    low_confidence: bool
    low_confidence_reasons: list[str]
    threshold: float
    model: dict


def _detector() -> Detector:
    if state["detector"] is None:
        raise HTTPException(503, f"No model loaded ({state['error']}). Train and export one, or copy model.onnx "
                                 f"and model_meta.json into the models folder.")
    return state["detector"]


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The image is larger than 25 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = ImageOps.exif_transpose(img)  # phone photos: honour the rotation flag, as a viewer would
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "The upload is not an image this service can read (use JPEG, PNG or TIFF).")
    return img.convert("RGB")


def _predict(det: Detector, img: Image.Image, conf: float | None) -> Prediction:
    thr = det.threshold if conf is None else conf
    rows = det.predict(img, thr)
    detections = [
        Detection(label=det.classes[int(r[5])], confidence=round(float(r[4]), 4), box=[round(float(v), 1) for v in r[:4]])
        for r in sorted(rows.tolist(), key=lambda r: -r[4])
    ]
    counts = {name: sum(1 for d in detections if d.label == name) for name in det.classes}
    reasons = []
    margin = float(det.meta.get("low_conf_margin", 0.15))
    borderline = sum(1 for d in detections if d.label == det.classes[0] and d.confidence < thr + margin)
    if borderline:
        reasons.append(f"{borderline} parasite detection(s) are close to the decision threshold")
    odd = quality.outside_range(quality.features(img), det.meta.get("quality_ranges", {}))
    if odd:
        reasons.append("image looks unlike the training images: " + ", ".join(odd))
    parasites, wbc = counts.get(det.classes[0], 0), counts.get("wbc", 0)
    return Prediction(
        image={"width": img.size[0], "height": img.size[1]},
        detections=detections,
        counts=counts,
        parasites_per_wbc=round(parasites / wbc, 3) if wbc else None,
        low_confidence=bool(reasons),
        low_confidence_reasons=reasons,
        threshold=thr,
        model={k: det.meta.get(k) for k in ("name", "trained_on", "imgsz", "exported")},
    )


@app.get("/health")
def health() -> dict:
    det = state["detector"]
    return {"status": "ok" if det else "no_model", "error": state["error"],
            "model": det.meta.get("name") if det else None}


@app.get("/model")
def model_info() -> dict:
    return _detector().meta


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...), conf: float | None = Query(None, gt=0, lt=1)) -> Prediction:
    det = _detector()
    return _predict(det, await _read_image(file), conf)


@app.post("/predict/annotated", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def predict_annotated(file: UploadFile = File(...), conf: float | None = Query(None, gt=0, lt=1)) -> Response:
    det = _detector()
    img = await _read_image(file)
    result = _predict(det, img, conf)
    draw = ImageDraw.Draw(img)
    width = max(2, img.size[0] // 500)
    for d in result.detections:
        draw.rectangle(d.box, outline=COLOURS.get(det.classes.index(d.label), (0, 0, 0)), width=width)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
