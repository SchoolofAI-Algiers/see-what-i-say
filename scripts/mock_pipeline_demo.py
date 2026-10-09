#!/usr/bin/env python3
"""Run a complete STARK VISION command-to-mask demo with deterministic mocks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stark_vision.mocks.pipeline import make_default_mock_pipeline


def main() -> None:
    pipeline = make_default_mock_pipeline()

    def issue_voice_command() -> None:
        step = pipeline.listen_and_dispatch()
        print(f"heard: {step.transcript}")
        print(f"parsed: {step.command.type.value}")
        print(f"status: {pipeline.manager.message}")
        if pipeline.manager.pending and pipeline.manager.pending.ambiguous:
            print(f"candidates: {len(pipeline.manager.pending.candidates)} (awaiting clarification)")
        if pipeline.manager.active:
            active = pipeline.manager.active
            print(
                f"active target #{active.target_id}: {active.prompt}; "
                f"mask={active.mask.shape if active.mask is not None else None}; "
                f"mask dtype={active.mask.dtype if active.mask is not None else None}"
            )

    pipeline.capture_next()  # Frame 0 is the exact grounding seed.
    issue_voice_command()

    # Camera/display continue while the request waits for clarification.
    for _ in range(4):
        pipeline.capture_next()

    issue_voice_command()  # Resolves the retained list; Grounder is not called again.
    pipeline.capture_next()
    issue_voice_command()  # Recolor the active mask.
    print(f"rendered RGB shape: {pipeline.last_rendered.image.shape}")
    pipeline.capture_next()
    issue_voice_command()  # A new effect replaces recolor.
    pipeline.capture_next()
    issue_voice_command()  # Clear target, tracker state, and effect.

    print(f"grounding calls: {len(pipeline.grounder.calls)} (expected 1)")
    print(f"tracker initializations: {pipeline.tracker.initialize_count} (expected 1)")
    print("mock pipeline demo completed")


if __name__ == "__main__":
    main()
