"""Application policy for selecting and maintaining the active target."""

from __future__ import annotations

from dataclasses import dataclass

from stark_vision.contracts import (
    CandidateSelector,
    CandidateSelectorKind,
    EffectMode,
    EffectSpec,
    GroundingCandidate,
    GroundingResult,
    TargetSnapshot,
    TrackingResult,
    TrackingStatus,
)


@dataclass
class PendingAcquisition:
    request_id: int
    prompt: str
    source_frame_id: int
    candidates: tuple[GroundingCandidate, ...] = ()
    selected: GroundingCandidate | None = None
    ambiguous: bool = False
    target_id: int | None = None


class TargetManager:
    """Keep active tracking stable while a replacement target is acquired."""

    def __init__(self, score_margin: float = 0.15) -> None:
        if score_margin < 0:
            raise ValueError("score_margin must be non-negative")
        self.score_margin = score_margin
        self.active: TargetSnapshot | None = None
        self.pending: PendingAcquisition | None = None
        self.effect = EffectSpec(EffectMode.NONE)
        self.message = "Ready"
        self._next_target_id = 1

    def begin_acquisition(self, request_id: int, prompt: str, source_frame_id: int) -> None:
        self.pending = PendingAcquisition(request_id, prompt, source_frame_id)
        self.message = f"Finding {prompt}"

    def update_seed_frame(self, request_id: int, source_frame_id: int) -> bool:
        if self.pending is None or self.pending.request_id != request_id:
            return False
        self.pending.source_frame_id = source_frame_id
        self.pending.candidates = ()
        self.pending.selected = None
        self.pending.ambiguous = False
        self.pending.target_id = None
        return True

    def receive_grounding_result(self, result: GroundingResult) -> GroundingCandidate | None:
        pending = self.pending
        if pending is None or pending.request_id != result.request_id:
            return None
        if pending.source_frame_id != result.source_frame_id:
            return None
        return self.receive_candidates(result.request_id, result.candidates)

    def receive_candidates(
        self, request_id: int, candidates: tuple[GroundingCandidate, ...] | list[GroundingCandidate]
    ) -> GroundingCandidate | None:
        pending = self.pending
        if pending is None or pending.request_id != request_id:
            return None
        ordered = tuple(sorted(candidates, key=lambda item: item.score, reverse=True))
        pending.candidates = ordered
        pending.selected = None
        pending.ambiguous = False
        pending.target_id = None
        if not ordered:
            self.pending = None
            self.message = "No matching target found"
            return None
        if len(ordered) > 1 and ordered[0].score - ordered[1].score < self.score_margin:
            pending.ambiguous = True
            self.message = "Several candidates found; say left, right, or a candidate number"
            return None
        return self._select(ordered[0])

    def resolve_candidate(self, selector: CandidateSelector | None) -> GroundingCandidate | None:
        pending = self.pending
        if pending is None or not pending.ambiguous or selector is None:
            return None
        chosen: GroundingCandidate | None = None
        if selector.kind is CandidateSelectorKind.RANK and isinstance(selector.value, int):
            if 1 <= selector.value <= len(pending.candidates):
                chosen = pending.candidates[selector.value - 1]
        elif selector.kind is CandidateSelectorKind.SIDE and selector.value in ("left", "right"):
            ordered = sorted(pending.candidates, key=lambda item: item.box.center[0])
            chosen = ordered[0] if selector.value == "left" else ordered[-1]
        return self._select(chosen) if chosen is not None else None

    def pending_target_id(self) -> int | None:
        return self.pending.target_id if self.pending else None

    def abort_pending(self, message: str) -> None:
        self.pending = None
        self.message = message

    def promote(self, result: TrackingResult) -> int | None:
        pending = self.pending
        if (
            pending is None
            or pending.target_id != result.target_id
            or result.mask is None
            or result.status is TrackingStatus.LOST
        ):
            return None
        old_id = self.active.target_id if self.active else None
        self.active = TargetSnapshot(
            target_id=result.target_id,
            prompt=pending.prompt,
            box=result.box,
            mask=result.mask,
            confidence=result.confidence,
            status=result.status,
            frame_id=result.frame_id,
        )
        self.pending = None
        self.message = f"Tracking {self.active.prompt}"
        return old_id

    def observe(self, results: tuple[TrackingResult, ...] | list[TrackingResult]) -> None:
        if self.active is None:
            return
        result = next((item for item in results if item.target_id == self.active.target_id), None)
        if result is None:
            return
        self.active = TargetSnapshot(
            target_id=result.target_id,
            prompt=self.active.prompt,
            box=result.box,
            mask=result.mask,
            confidence=result.confidence,
            status=result.status,
            frame_id=result.frame_id,
        )
        self.message = "Target lost" if result.status is TrackingStatus.LOST else f"Tracking {self.active.prompt}"

    def set_effect(self, effect: EffectSpec) -> bool:
        if self.active is None:
            self.message = "Select a target before applying an effect"
            return False
        self.effect = effect
        self.message = f"Effect: {effect.mode.value}"
        return True

    def stop(self) -> tuple[int, ...]:
        removed = tuple(
            target_id for target_id in (
                self.active.target_id if self.active else None,
                self.pending.target_id if self.pending else None,
            ) if target_id is not None
        )
        self.active = None
        self.pending = None
        self.effect = EffectSpec(EffectMode.NONE)
        self.message = "Tracking stopped"
        return removed

    def _select(self, candidate: GroundingCandidate) -> GroundingCandidate:
        assert self.pending is not None
        self.pending.selected = candidate
        if self.pending.target_id is None:
            self.pending.target_id = self._next_target_id
            self._next_target_id += 1
        self.message = f"Candidate selected: {candidate.label}"
        return candidate
