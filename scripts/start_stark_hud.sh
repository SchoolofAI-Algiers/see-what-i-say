#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=========================================================="
echo "      STARK VISION // MARK LXXXV AR HUD COCKPIT           "
echo "=========================================================="
echo " [1/2] Launching FastAPI Grounding Engine (Port 8000)..."
"$DIR/.venv/bin/python" -m uvicorn stark_vision.api.server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo " [2/2] Launching Next.js 14 HUD Cockpit (Port 3000)..."
cd "$DIR/frontend"
npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT INT TERM

echo ""
echo " >> STARK VISION HUD ONLINE:"
echo "    - Cockpit:   http://localhost:3000"
echo "    - API Docs:  http://localhost:8000/docs"
echo "    - WebSocket: ws://localhost:8000/api/ws/ground"
echo ""

wait
