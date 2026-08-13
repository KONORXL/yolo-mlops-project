from pathlib import Path
from typing import Any
import io

from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Response
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path("models/best.pt")

app = FastAPI(
    title="PPE Detection API",
    description="API for PPE object detection using a trained YOLOv8 model",
    version="1.0.0",
)

model = None

@app.get("/")
def root() -> dict[str, str]:
    return {"message": "PPE Detection API is running"}

@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0)
) -> dict[str, Any]:

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    results = model.predict(image, conf=conf, verbose=False)

    detections = []
    result = results[0]

    if result.boxes is not None:
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": names[cls_id],
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": [round(v, 2) for v in xyxy],
                }
            )

    return {
        "filename": file.filename,
        "num_detections": len(detections),
        "detections": sorted(detections, key=lambda x: x["confidence"], reverse=True),
    }
    
@app.post(
    "/predict-image",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Annotated image with detections",
        }
    },
)
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
) -> Response:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    results = model.predict(image, conf=conf, verbose=False)
    result = results[0]

    # Ultralytics devuelve la imagen anotada como array NumPy (BGR/RGB según backend),
    # y aquí la convertimos a JPEG en memoria para devolverla directamente.
    plotted = result.plot()

    annotated_image = Image.fromarray(plotted)
    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/jpeg")