"""FastAPI application factory."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

from ..alert.decision import TemporalDecision
from ..core.config import load_app_config
from ..models.yolo_detector import YOLODetector


def create_app(config_path: str = "configs/app.yaml") -> FastAPI:
    """Build the FastAPI app from a config file."""
    cfg = load_app_config(config_path)
    detector = YOLODetector(
        cfg.model.weights, cfg.model.conf, cfg.model.iou, device=cfg.model.device
    )
    decision = TemporalDecision(cfg.decision.window, cfg.decision.min_hits)

    app = FastAPI(title="Forest Fire Smoke Detection API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/detect")
    # File(...) 是 FastAPI 官方惯例写法，B008 在此为误报
    async def detect(file: UploadFile = File(...)) -> dict[str, object]:  # noqa: B008
        raw = await file.read()
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "invalid image"}

        detections = detector.predict(frame)
        alarming = decision.update(len(detections) > 0)
        return {
            "alarming": alarming,
            "count": len(detections),
            "detections": [
                {"cls": d.cls_name, "conf": round(d.confidence, 4), "xyxy": d.xyxy}
                for d in detections
            ],
        }

    return app
