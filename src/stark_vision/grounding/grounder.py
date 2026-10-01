"""
Grounding Engine for STARK VISION.

Provides open-vocabulary referring expression grounding via Grounding DINO.
Decoupled from audio/speech recognition and downstream mask segmentation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundingCandidate:
    """
    Immutable representation of a grounded visual object candidate.

    Attributes:
        box: Coordinates [x1, y1, x2, y2] in absolute display pixels.
        re: Mscoultimodal alignment confidence score in [0.0, 1.0].
        label: The matched referring expression phrase.
    """

    box: tuple[float, float, float, float]
    score: float
    label: str

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


class Grounder:
    """
    Production-grade Grounding DINO wrapper for referring expression grounding.

    Takes camera frames (BGR) and natural language text expressions, performs
    cross-modality feature fusion, and predicts spatial bounding boxes for
    the described entities.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> None:
        """
        Initialize the Grounder engine.

        Args:
            model_id: HuggingFace model repo or local directory path.
            device: 'cuda', 'cpu', or None (auto-detects CUDA).
            box_threshold: Minimum confidence threshold for bounding box existence.
            text_threshold: Minimum cross-modal similarity threshold for token alignment.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model_id = model_id

        logger.info(
            "Initializing Grounding DINO on device '%s' (model: %s)",
            self.device,
            self.model_id,
        )

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()
        load_time = (time.perf_counter() - t0) * 1000

        logger.info("Grounding DINO initialized in %.1f ms", load_time)

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        """
        Normalize text query for Grounding DINO tokenizer across all edge cases.

        Handles:
        - Case insensitivity (lowercasing)
        - Removing leading/trailing punctuation and redundant spaces
        - Splitting comma/semicolon/slash separated expressions into period-separated phrases
        - Ensuring every distinct sub-phrase ends with a trailing '.'
        """
        import re

        cleaned = text.strip().lower()
        if not cleaned:
            return "."

        # Replace separators (comma, semicolon, slash, pipe) with period delimiter
        cleaned = re.sub(r"[,;/|]+", " . ", cleaned)

        # Normalize multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Split on existing periods and reconstruct clean dot-separated phrases
        phrases = [p.strip() for p in cleaned.split(".") if p.strip()]
        if not phrases:
            return "."

        # Append trailing dot to every phrase as required by Grounding DINO BERT tokenizer
        return " . ".join(phrases) + " ."

    def find_candidates(
        self,
        frame_bgr: np.ndarray,
        text_prompt: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[GroundingCandidate]:
        """
        Ground a text query into bounding box candidates within a camera frame.

        Args:
            frame_bgr: OpenCV image array in BGR format (H, W, 3).
            text_prompt: Natural language query (e.g., "guy wearing red shirt", "tv box").
            box_threshold: Optional override for box confidence threshold.
            text_threshold: Optional override for text alignment threshold.

        Returns:
            List of GroundingCandidate instances sorted by score descending.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame_bgr cannot be None or empty.")

        if not text_prompt or not text_prompt.strip():
            logger.debug("Empty text prompt provided; returning 0 candidates.")
            return []

        query = self._normalize_prompt(text_prompt)
        b_thresh = box_threshold if box_threshold is not None else self.box_threshold
        t_thresh = text_threshold if text_threshold is not None else self.text_threshold

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)


        inputs = self.processor(
            images=pil_image, text=query, return_tensors="pt"
        ).to(self.device)

        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs to absolute pixel coordinates [x1, y1, x2, y2]
        target_sizes = [(h, w)]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=b_thresh,
            text_threshold=t_thresh,
            target_sizes=target_sizes,
        )[0]
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        # Retrieve text labels if available, fallback to query string
        labels = results.get("text_labels") or results.get("labels", [])

        candidates: list[GroundingCandidate] = []
        for i in range(len(boxes)):
            box_coords = (
                float(boxes[i][0]),
                float(boxes[i][1]),
                float(boxes[i][2]),
                float(boxes[i][3]),
            )
            score_val = float(scores[i])
            label_val = str(labels[i]) if i < len(labels) else query.rstrip(" .")

            candidate = GroundingCandidate(
                box=box_coords,
                score=score_val,
                label=label_val,
            )
            candidates.append(candidate)

        # Sort by confidence descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        logger.debug(
            "Grounding query '%s' found %d candidates in %.1f ms",
            query,
            len(candidates),
            elapsed_ms,
        )
        return candidates

    def find(
        self,
        frame_bgr: np.ndarray,
        text_prompt: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[GroundingCandidate]:
        """
        Primary interface contract matching team specification: grounder.find().

        Args:
            frame_bgr: OpenCV image array in BGR format (H, W, 3).
            text_prompt: Referring expression (e.g., 'guy wearing red shirt').
            box_threshold: Optional override for box confidence threshold.
            text_threshold: Optional override for text alignment threshold.

        Returns:
            List of GroundingCandidate instances sorted by score descending.
        """
        return self.find_candidates(
            frame_bgr=frame_bgr,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    @staticmethod
    def is_ambiguous(
        candidates: Sequence[GroundingCandidate],
        delta_threshold: float = 0.15,
    ) -> bool:
        """
        Convenience wrapper for candidate ambiguity check.
        """
        return is_ambiguous(candidates, delta_threshold=delta_threshold)


def is_ambiguous(
    candidates: Sequence[GroundingCandidate],
    delta_threshold: float = 0.15,
) -> bool:
    """
    Evaluate whether visual detection candidates are ambiguous.

    If two or more candidates have confidence scores that are within
    `delta_threshold` of each other, the target cannot be determined
    unambiguously without user clarification.

    Args:
        candidates: Sequence of GroundingCandidate sorted by score descending.
        delta_threshold: Minimum score delta required to declare a clear winner.

    Returns:
        True if candidates are ambiguous, False otherwise.
    """
    if len(candidates) < 2:
        return False
    score_gap = abs(candidates[0].score - candidates[1].score)
    return score_gap < delta_threshold

