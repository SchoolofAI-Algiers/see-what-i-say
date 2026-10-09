# STARK VISION Setup & Integration Guide

Step-by-step commands to install dependencies, run tests, and integrate the visual grounding engine.

## 1. Environment Setup

```bash
# Clone and enter directory
cd see-what-i-say

# Install Python backend dependencies using uv
uv sync

# Install Next.js frontend dependencies (for HUD cockpit)
cd frontend
npm install
cd ..
```

## 2. Running Verification Tests

Run the full unit and contract compliance test suite:

```bash
# Runs all 51 tests across grounding, contracts, geometry, HUD, and API
uv run pytest tests/ -v
```

## 3. Running the Live Application

### Option A: Unified Launcher (FastAPI Backend + Next.js Cockpit)
Starts the Grounding DINO Base API on port 8000 and the Next.js HUD Cockpit on port 3000:

```bash
./scripts/start_stark_hud.sh
```
- Cockpit UI: http://localhost:3000
- API Docs: http://localhost:8000/docs
- WebSocket Endpoint: `ws://localhost:8000/api/ws/ground`

### Option B: Running Standalone Components

**1. FastAPI Grounding Engine:**
```bash
PYTHONPATH=src uv run uvicorn stark_vision.api.server:app --host 0.0.0.0 --port 8000
```

**2. Next.js HUD Cockpit:**
```bash
cd frontend && npm run dev
```

**3. Headless / Direct OpenCV Live Runner:**
```bash
uv run python scripts/test_grounder_live.py
```

## 4. Subsystem Integration Code Example

How teammates (Orchestration / SAM 2) call Grounding directly in Python:

```python
import numpy as np
from stark_vision.grounding import Grounder, is_ambiguous
from stark_vision.contracts import Frame, BoundingBox

# Initialize Grounder (auto-detects CUDA FP16)
grounder = Grounder(model_id="IDEA-Research/grounding-dino-base")

# Option 1: Ingesting Rayan's Frame contract (RGB uint8)
frame = Frame(id=1, timestamp_ns=0, image=np.zeros((720, 1280, 3), dtype=np.uint8))
candidates = grounder.find(frame, "the laptop on the table")

# Option 2: Ingesting raw OpenCV frame (BGR or RGB)
# candidates = grounder.find(raw_bgr_frame, "the person on the left")

for cand in candidates:
    print(f"Target: {cand.label}")
    print(f"Confidence: {cand.score:.3f}")
    print(f"BoundingBox: x1={cand.box.x1}, y1={cand.box.y1}, x2={cand.box.x2}, y2={cand.box.y2}")
    print(f"Center for sorting: {cand.box.center}")  # Supports cand.box.center[0]

# Check if candidates require voice clarification
ambiguous = is_ambiguous(candidates, text_prompt="the laptop")
```
