"""
    script for testing in dev : run on a video to a video instead of live cam 
    this can run on collab T4
    to run :
        uv run python scripts/run_on_video.py input.mp4 out.mp4 \
            --command "reffering expression" --command "action"
    to run mock pipeline and components ( test only does nothing ):
        uv run python scripts/run_on_video.py input.mp4 out.mp4 --mock --command "segment the person"
"""
from __future__ import annotations
import argparse
import logging

from stark_vision.main import App
from stark_vision.utils.logging_setup import setup_logging
from stark_vision.video import process_video

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--command", action="append", default=[], help="repeatable; issued at --command-frame")
    p.add_argument("--command-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--mock", action="store_true", help="use mock grounder/tracker (no models)")
    p.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--sam-checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    setup_logging()

    if a.mock:
        # import here to save time
        from stark_vision.mocks.adapters import MockGrounder, MockTracker
        grounder, tracker = MockGrounder(), MockTracker()
    else:
        from stark_vision.grounding.grounder import Grounder
        from stark_vision.segmentation.sam2_wrapper import SAM2Tracker

        grounder = Grounder(model_id=a.dino_model)
        tracker = SAM2Tracker(a.sam_config, a.sam_checkpoint, device=a.device, window_size=a.window)
    app = App(grounder, tracker)
    try:
        stats = process_video(a.src, a.dst, app, a.command, command_frame=a.command_frame, max_frames=a.max_frames)
    finally:
        app.close()
    logging.getLogger("run_on_video").info("done: %s", stats)

if __name__ == "__main__":
    main()