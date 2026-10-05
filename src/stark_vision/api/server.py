"""
STARK VISION - FastAPI WebSocket Streaming Service
Bridges Next.js 14 HUD Cockpit with CUDA Grounding DINO Engine.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stark_vision.grounding.grounder import Grounder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stark_vision.api")

app = FastAPI(
    title="STARK VISION API",
    version="2.0.0",
    description="High-concurrency multimodal referring video segmentation server",
)

# Enable CORS for Next.js dev server on port 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Grounder instance
_grounder: Grounder | None = None


def get_grounder() -> Grounder:
    global _grounder
    if _grounder is None:
        model_id = "IDEA-Research/grounding-dino-base"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing Grounder singleton (%s on %s)...", model_id, device)
        _grounder = Grounder(model_id=model_id, device=device)
    return _grounder


class HealthResponse(BaseModel):
    status: str
    engine: str
    device: str
    vram_allocated_mb: float


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    return HealthResponse(
        status="ONLINE",
        engine="Grounding DINO Base (Swin-T)",
        device=device,
        vram_allocated_mb=vram_mb,
    )


@app.get("/api/telemetry")
def get_telemetry() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_mb = (
        round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0
    )
    vram_total = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0
    )
    return {
        "status": "ONLINE",
        "gpu": gpu_name,
        "vram_used_mb": vram_mb,
        "vram_total_mb": vram_total,
        "framework": "PyTorch + FastAPI",
    }


def decode_base64_image(base64_str: str) -> np.ndarray | None:
    """Decode base64 data URL into OpenCV BGR numpy array."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.warning("Failed to decode base64 frame: %s", e)
        return None


@app.websocket("/api/ws/ground")
async def websocket_grounding_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint streaming frames from Next.js and returning detected target reticles.
    """
    await websocket.accept()
    logger.info("WebSocket connection established with client: %s", websocket.client)

    grounder = get_grounder()

    try:
        while True:
            raw_text = await websocket.receive_text()
            data = json.loads(raw_text)

            msg_type = data.get("type", "ground_frame")
            if msg_type in ("ground_frame", "update_query"):
                query = data.get("query", "person").strip()
                box_thresh = float(data.get("box_threshold", 0.35))
                text_thresh = float(data.get("text_threshold", 0.25))
                img_data = data.get("image_data")

                detections_list: list[dict[str, Any]] = []
                inference_ms = 0.0

                if img_data:
                    frame = decode_base64_image(img_data)
                    if frame is not None and query:
                        h, w = frame.shape[:2]
                        t0 = time.perf_counter()
                        try:
                            candidates = grounder.find(
                                frame_bgr=frame,
                                text_prompt=query,
                                box_threshold=box_thresh,
                                text_threshold=text_thresh,
                            )
                            inference_ms = round((time.perf_counter() - t0) * 1000, 1)

                            is_ambiguous = len(candidates) > 1

                            for idx, c in enumerate(candidates, start=1):
                                # Normalize coordinates to [0.0, 1.0]
                                x1_norm = max(0.0, min(1.0, c.x1 / w))
                                y1_norm = max(0.0, min(1.0, c.y1 / h))
                                x2_norm = max(0.0, min(1.0, c.x2 / w))
                                y2_norm = max(0.0, min(1.0, c.y2 / h))

                                box_height_norm = max(0.01, y2_norm - y1_norm)
                                # Empirical depth estimation heuristic: d ~ 1.5 / height_ratio
                                est_distance = round(1.2 + (1.0 - box_height_norm) * 2.2, 2)

                                detections_list.append(
                                    {
                                        "id": idx,
                                        "label": c.label.upper(),
                                        "confidence": round(c.score, 3),
                                        "box": [
                                            round(x1_norm, 4),
                                            round(y1_norm, 4),
                                            round(x2_norm, 4),
                                            round(y2_norm, 4),
                                        ],
                                        "distance": est_distance,
                                        "isAmbiguous": is_ambiguous,
                                    }
                                )
                        except Exception as ex:
                            logger.error("Error during Grounding DINO execution: %s", ex)

                # Query GPU memory
                vram_used = 0.0
                if torch.cuda.is_available():
                    vram_used = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)

                response_payload = {
                    "type": "ground_result",
                    "query": query,
                    "detections": detections_list,
                    "inference_ms": inference_ms,
                    "vram_used_mb": vram_used,
                    "timestamp": time.time(),
                }
                await websocket.send_text(json.dumps(response_payload))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected gracefully.")
    except Exception as e:
        logger.warning("WebSocket loop exception: %s", e)
