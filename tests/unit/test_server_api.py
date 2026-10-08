"""
Unit tests for STARK VISION FastAPI server endpoints and WebSocket handlers.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stark_vision.api.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Verify health endpoint returns status, engine, and camera state."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "Grounding DINO" in data["engine"]
    assert "camera_active" in data


def test_telemetry_endpoint(client):
    """Verify telemetry returns GPU, VRAM, and camera telemetry."""
    response = client.get("/api/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "vram_used_mb" in data
    assert "camera" in data
    assert "fps" in data["camera"]


@pytest.mark.anyio
async def test_stream_mjpeg_generator():
    """Verify stream_mjpeg async generator yields multipart chunks."""
    from stark_vision.api.server import stream_mjpeg
    gen = stream_mjpeg()
    chunk = await gen.__anext__()
    assert b"--frame" in chunk
    assert b"Content-Type: image/jpeg" in chunk
    await gen.aclose()


def test_video_feed_endpoint(client, monkeypatch):
    """Verify video feed route serves multipart MJPEG streaming."""
    async def mock_stream():
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfakejpeg\r\n"

    monkeypatch.setattr("stark_vision.api.server.stream_mjpeg", mock_stream)
    with client.stream("GET", "/api/video_feed") as response:
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers.get("content-type", "")
        chunks = list(response.iter_bytes())
        assert len(chunks) > 0


def test_websocket_ping_pong(client):
    """Verify WebSocket responds to ping messages."""
    with client.websocket_connect("/api/ws/ground") as ws:
        ws.send_json({"type": "ping"})
        msg = ws.receive_json()
        assert msg["type"] == "pong"
        assert "timestamp" in msg


def test_ground_empty_query(client):
    """Verify handling of empty grounding query returns safe empty detections."""
    response = client.post("/api/ground", json={"query": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["detections"] == []
    assert data["query"] == ""
