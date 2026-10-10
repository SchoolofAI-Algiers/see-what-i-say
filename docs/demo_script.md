# STARK VISION Live Demo Script

Guide for demonstrating the grounding engine and Mark LXXXV HUD during defense presentations.

## Pre-Flight Check

1. Connect webcam (`/dev/video0`).
2. Verify ports 8000 and 3000 are free:
   ```bash
   fuser 8000/tcp 3000/tcp /dev/video0
   ```
3. Run test suite to verify CUDA execution:
   ```bash
   uv run pytest tests/unit/test_contracts_compliance.py -v
   ```

## Launching the Demo

Run the unified launcher:
```bash
./scripts/start_stark_hud.sh
```

Wait ~5 seconds for model pre-load:
```text
Grounding DINO initialized in ~4200 ms
Grounding DINO Base preloaded and warmed up in VRAM.
Uvicorn running on http://0.0.0.0:8000
Next.js ready on http://localhost:3000
```

Open browser at `http://localhost:3000`.

## Interactive Demonstration Steps

1. **Boot Diagnostic Sequence:**
   - On page load, observe the 2.5-second Mark LXXXV boot diagnostics, rotating Arc Reactor wireframe, and HUD visor bracket calibration.
2. **Simple Query Test (High Threshold Baseline):**
   - Click `#person` chip or type `person` and press Enter.
   - Observe immediate target lock (`confidence ~0.65+`, latency ~85ms) with distance readout.
3. **Complex Descriptive Query (Dynamic Adaptive Thresholding):**
   - Type a longer descriptive query: `laptop on desk` or `person wearing glasses`.
   - Explain to mentors that DAT automatically lowered the threshold to capture the multi-word referent without triggering false alarms.
4. **Spatial Disambiguation Test:**
   - Type `the person on the left` or `the one on the right`.
   - Observe automatic spatial sorting resolving the correct candidate without pausing for clarification.
5. **Jitter Stability Check:**
   - Point out that reticle brackets remain locked on the target without shaking due to the 1-Euro centroid filter.
