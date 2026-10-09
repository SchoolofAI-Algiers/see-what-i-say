"""
Simple launcer script.
Can spin up just the fastapi server, run the quick live test, or kick off the full stack.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stark Vision launcher")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run fastapi websocket backend only",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run local OpenCV grounding diagnostic test",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.server:
        import uvicorn
        print(">> Launching STARK VISION FastAPI Server on port 8000...")
        uvicorn.run("stark_vision.api.server:app", host="0.0.0.0", port=8000, reload=True)
    elif args.test:
        test_script = project_root / "scripts" / "test_grounder_live.py"
        subprocess.run([sys.executable, str(test_script)])
    else:
        # Default: launch full HUD stack
        start_script = project_root / "scripts" / "start_stark_hud.sh"
        if start_script.exists():
            subprocess.run(["bash", str(start_script)])
        else:
            import uvicorn
            uvicorn.run("stark_vision.api.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
