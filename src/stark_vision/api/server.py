"""
STARK VISION - FastAPI Hybrid Streaming & Grounding Service.

Decouples camera ingestion and model execution:
- Python captures native webcam at 30 FPS via CameraManager (buffer size 1).
- Serves low-overhead MJPEG stream at /api/video_feed.
- Executes Grounding DINO Base on-demand via WebSocket or REST.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from stark_vision.api.camera_manager import CameraManager
from stark_vision.geometry.coords import normalize_box
from stark_vision.grounding.grounder import Grounder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stark_vision.api")

# Singletons
_grounder: Grounder | None = None
_camera_manager: CameraManager | None = None


def get_grounder() -> Grounder:
    """Retrieve or initialize the Grounding DINO Base engine."""
    global _grounder
    if _grounder is None:
        model_id = "IDEA-Research/grounding-dino-base"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing Grounder singleton (%s on %s)...", model_id, device)
        _grounder = Grounder(
            model_id=model_id,
            device=device,
            use_fp16=True,
            enable_temporal_filter=True,
        )
    return _grounder


def get_camera_manager() -> CameraManager:
    """Retrieve or initialize the CameraManager singleton."""
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager(device_id=0, target_width=640, target_height=480, target_fps=30)
        _camera_manager.start()
    return _camera_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context for hardware initialization and cleanup."""
    logger.info("Starting STARK VISION API Server...")
    cam = get_camera_manager()
    yield
    logger.info("Shutting down STARK VISION API Server...")
    if cam:
        cam.stop()


app = FastAPI(
    title="STARK VISION API",
    version="2.0.0",
    description="Referring expression grounding and real-time vision telemetry engine",
    lifespan=lifespan,
)

# Enable CORS for Next.js frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    engine: str
    device: str
    vram_allocated_mb: float
    camera_active: bool


class GroundRequest(BaseModel):
    query: str
    box_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    text_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    image_data: str | None = None


def decode_base64_image(base64_str: str) -> np.ndarray | None:
    """Decode base64 data URL into OpenCV BGR ndarray."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning("Failed to decode base64 frame: %s", e)
        return None


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report service health, model runtime, and camera status."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    cam = get_camera_manager()
    cam_status = cam.get_status()
    return HealthResponse(
        status="ONLINE",
        engine="Grounding DINO Base (Swin-T, FP16)",
        device=device,
        vram_allocated_mb=vram_mb,
        camera_active=cam_status.get("hardware_active", False),
    )


@app.get("/api/telemetry")
def get_telemetry() -> dict[str, Any]:
    """Detailed hardware and pipeline throughput metrics."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_used = (
        round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0.0
    )
    vram_total = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0.0
    )
    cam = get_camera_manager()
    return {
        "status": "ONLINE",
        "gpu": gpu_name,
        "vram_used_mb": vram_used,
        "vram_total_mb": vram_total,
        "camera": cam.get_status(),
        "framework": "PyTorch + FastAPI",
    }


async def stream_mjpeg() -> AsyncGenerator[bytes, None]:
    """Asynchronous generator yielding MJPEG multipart frames."""
    cam = get_camera_manager()
    try:
        while True:
            jpeg = cam.get_jpeg()
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.033)
    except (asyncio.CancelledError, GeneratorExit):
        pass


@app.get("/api/video_feed")
async def video_feed() -> StreamingResponse:
    """Stream 30 FPS MJPEG video directly to HTML <img> tag or canvas."""
    return StreamingResponse(
        stream_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/ground")
def ground_endpoint(req: GroundRequest) -> dict[str, Any]:
    """On-demand REST grounding execution on the latest video frame."""
    query = req.query.strip()
    if not query:
        return {"query": "", "detections": [], "inference_ms": 0.0, "vram_used_mb": 0.0}

    cam = get_camera_manager()
    frame = decode_base64_image(req.image_data) if req.image_data else cam.get_frame()

    grounder = get_grounder()
    t0 = time.perf_counter()
    candidates = grounder.find(
        frame_bgr=frame,
        text_prompt=query,
        box_threshold=req.box_threshold,
        text_threshold=req.text_threshold,
    )
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    h, w = frame.shape[:2]
    is_ambig = grounder.is_ambiguous(candidates, text_prompt=query)
    detections: list[dict[str, Any]] = []

    for idx, c in enumerate(candidates, start=1):
        x1_n, y1_n, x2_n, y2_n = normalize_box(c.box, width=w, height=h)
        box_h_n = max(0.01, y2_n - y1_n)
        est_distance = round(1.2 + (1.0 - box_h_n) * 2.2, 2)
        detections.append(
            {
                "id": idx,
                "label": c.label.upper(),
                "confidence": round(c.score, 3),
                "box": [round(x1_n, 4), round(y1_n, 4), round(x2_n, 4), round(y2_n, 4)],
                "distance": est_distance,
                "isAmbiguous": is_ambig,
            }
        )

    vram_used = (
        round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0.0
    )

    return {
        "query": query,
        "detections": detections,
        "inference_ms": inference_ms,
        "vram_used_mb": vram_used,
        "timestamp": time.time(),
    }


@app.websocket("/api/ws/ground")
async def websocket_grounding_endpoint(websocket: WebSocket) -> None:
    """
    Bi-directional event-driven WebSocket for real-time interactive HUD grounding.
    Executes Grounding DINO on-demand without unbounded frame queueing.
    """
    await websocket.accept()
    logger.info("Client connected to /api/ws/ground: %s", websocket.client)

    grounder = get_grounder()
    cam = get_camera_manager()

    try:
        while True:
            raw_text = await websocket.receive_text()
            data = json.loads(raw_text)

            msg_type = data.get("type", "ground")

            if msg_type in ("ground", "update_query", "ground_frame"):
                query = data.get("query", "").strip()
                b_thresh = data.get("box_threshold")
                t_thresh = data.get("text_threshold")
                img_data = data.get("image_data")

                if b_thresh is not None:
                    b_thresh = float(b_thresh)
                if t_thresh is not None:
                    t_thresh = float(t_thresh)

                detections_list: list[dict[str, Any]] = []
                inference_ms = 0.0

                if query:
                    # Select frame: client snapshot or native OpenCV camera frame
                    frame = decode_base64_image(img_data) if img_data else cam.get_frame()
                    h, w = frame.shape[:2]

                    t0 = time.perf_counter()
                    try:
                        candidates = grounder.find(
                            frame_bgr=frame,
                            text_prompt=query,
                            box_threshold=b_thresh,
                            text_threshold=t_thresh,
                        )
                        inference_ms = round((time.perf_counter() - t0) * 1000, 1)
                        is_ambig = grounder.is_ambiguous(candidates, text_prompt=query)

                        for idx, c in enumerate(candidates, start=1):
                            x1_n, y1_n, x2_n, y2_n = normalize_box(c.box, width=w, height=h)
                            box_h_n = max(0.01, y2_n - y1_n)
                            est_distance = round(1.2 + (1.0 - box_h_n) * 2.2, 2)

                            detections_list.append(
                                {
                                    "id": idx,
                                    "label": c.label.upper(),
                                    "confidence": round(c.score, 3),
                                    "box": [
                                        round(x1_n, 4),
                                        round(y1_n, 4),
                                        round(x2_n, 4),
                                        round(y2_n, 4),
                                    ],
                                    "distance": est_distance,
                                    "isAmbiguous": is_ambig,
                                }
                            )
                    except Exception as ex:
                        logger.error("Error during Grounding DINO execution: %s", ex)

                vram_used = (
                    round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
                    if torch.cuda.is_available()
                    else 0.0
                )

                response_payload = {
                    "type": "ground_result",
                    "query": query,
                    "detections": detections_list,
                    "inference_ms": inference_ms,
                    "vram_used_mb": vram_used,
                    "timestamp": time.time(),
                }
                await websocket.send_text(json.dumps(response_payload))

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected gracefully.")
    except Exception as e:
        logger.warning("WebSocket exception: %s", e)
