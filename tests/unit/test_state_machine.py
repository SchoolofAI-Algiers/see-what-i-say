"""Unit tests for TargetStateMachine, TargetManager, and rate-decoupled scheduling."""

import pytest
from stark_vision.target.state_machine import TargetStateMachine
from stark_vision.target.target_manager import TargetManager
from stark_vision.target.target_state import TargetState


def test_fsm_lifecycle_transitions():
    fsm = TargetStateMachine(recovery_confidence_threshold=0.40)
    assert fsm.state == TargetState.IDLE

    # 1. Trigger search
    fsm.trigger_query("guy in red shirt")
    assert fsm.state == TargetState.SEARCHING

    # 2. Grounding DINO finds 1 candidate
    fsm.on_grounding_candidates(candidate_count=1, is_ambiguous=False)
    assert fsm.state == TargetState.GROUNDED

    # 3. SAM 2 initiates tracking
    fsm.on_tracking_started()
    assert fsm.state == TargetState.TRACKING

    # 4. SAM 2 updates high confidence -> remains TRACKING
    fsm.on_tracker_frame(tracker_confidence=0.88)
    assert fsm.state == TargetState.TRACKING

    # 5. Tracker confidence drops below 0.40 -> RECOVERY
    fsm.on_tracker_frame(tracker_confidence=0.32)
    assert fsm.state == TargetState.RECOVERY

    # 6. Target lost
    fsm.on_target_lost()
    assert fsm.state == TargetState.LOST


def test_fsm_ambiguity_flow():
    fsm = TargetStateMachine()
    fsm.trigger_query("cup")
    assert fsm.state == TargetState.SEARCHING

    # Two matching cups detected
    fsm.on_grounding_candidates(
        candidate_count=2,
        is_ambiguous=True,
        clarification_msg="Detected 2 cups. Please specify left or right.",
    )
    assert fsm.state == TargetState.AMBIGUOUS
    assert fsm.clarification_message is not None


def test_rate_decoupling_scheduler():
    manager = TargetManager(grounding_cadence_sec=0.50, recovery_threshold=0.40)

    # Initially IDLE
    assert not manager.should_run_grounding()

    # User speaks prompt
    manager.set_query("laptop on desk")
    # SEARCHING must trigger immediate grounding
    assert manager.should_run_grounding(current_time=1.0)

    # Grounding completes and target initialized
    manager.mark_grounding_executed(current_time=1.0)
    manager.update_target(box=(100, 100, 200, 200), confidence=0.90, label="laptop", current_time=1.0)
    assert manager.state == TargetState.TRACKING

    # Next frame at 33ms later: should NOT run grounding (leaves GPU for SAM 2)
    assert not manager.should_run_grounding(current_time=1.033)

    # 550ms later: cadence elapsed -> triggers asynchronous re-verification
    assert manager.should_run_grounding(current_time=1.55)

    # If tracking drops into RECOVERY, must immediately run grounding
    manager.on_tracker_step(confidence=0.25, box=(100, 100, 200, 200))
    assert manager.state == TargetState.RECOVERY
    assert manager.should_run_grounding(current_time=1.56)
