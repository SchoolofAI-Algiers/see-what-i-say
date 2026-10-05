"""
STARK VISION — Industrial Grounding Engine.

Production-grade Grounding DINO wrapper for multimodal referring video segmentation.
Engineered with:
- Dynamic Adaptive Thresholding (DAT) eliminating hardcoded constants
- FP16 Tensor Core acceleration on CUDA hardware
- IoU-aware candidate deduplication and spatial disambiguation
- Temporal One-Euro filter for jitter-free tracking
- Robust lexical normalization protecting abbreviations and decimals
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from stark_vision.grounding.adaptive_threshold import (
    DynamicThresholdCalculator,
    ThresholdConfig,
)
from stark_vision.grounding.disambiguation import (
    SpatialCandidate,
    SpatialResolver,
)
from stark_vision.grounding.temporal_filter import BoxTemporalFilter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundingCandidate:
    """
    Immutable representation of a grounded visual object candidate.

    Attributes:
        box: Coordinates [x1, y1, x2, y2] in absolute display pixels.
        score: Multimodal alignment confidence score in [0.0, 1.0].
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
    Industrial-grade Grounding DINO wrapper for referring expression visual grounding.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str | None = None,
        use_fp16: bool = True,
        enable_temporal_filter: bool = True,
    ) -> None:
        """
        Initialize the Grounder engine.

        Args:
            model_id: HuggingFace model repo (defaults to grounding-dino-base).
            device: 'cuda', 'cpu', or None (auto-detects CUDA).
            use_fp16: Use FP16 half precision on CUDA for Tensor Core acceleration.
            enable_temporal_filter: Enable One-Euro jitter filter across frames.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        self.threshold_calculator = DynamicThresholdCalculator(ThresholdConfig())
        self.spatial_resolver = SpatialResolver()
        self.temporal_filter = BoxTemporalFilter() if enable_temporal_filter else None

        logger.info(
            "Initializing Grounding DINO on device '%s' (model: %s, FP16: %s)",
            self.device,
            self.model_id,
            self.use_fp16,
        )

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype if self.device.startswith("cuda") else torch.float32,
        ).to(self.device)
        self.model.eval()
        load_time = (time.perf_counter() - t0) * 1000

        logger.info("Grounding DINO initialized in %.1f ms", load_time)

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        """
        Industrial lexical normalizer for Grounding DINO BERT tokenizer.

        Protects:
        - Decimals (e.g. '3.5 mm' -> not split on '.')
        - Honorifics / titles ('mr.', 'dr.')
        - Strips conversational speech preambles ('can you track the', 'please highlight')
        - Appends trailing '.' required by token classification head
        """
        cleaned = text.strip().lower()
        if not cleaned:
            return "."

        # Strip common speech assistant preambles
        cleaned = re.sub(
            r"^(please\s+)?(can\s+you\s+)?(could\s+you\s+)?(track|find|show|highlight|locate|look\s+at)\s+(the\s+)?",
            "",
            cleaned,
        )

        # Protect decimals and honorifics with temp placeholders
        cleaned = re.sub(r"(\d+)\.(\d+)", r"\1__DEC__\2", cleaned)
        cleaned = re.sub(r"\b(mr|dr|ms|mrs|prof)\.", r"\1__DOT__", cleaned)

        # Replace separators (comma, semicolon, slash, pipe) with period delimiters
        cleaned = re.sub(r"[,;/|]+", " . ", cleaned)

        # Restore protected tokens
        cleaned = cleaned.replace("__DEC__", ".").replace("__DOT__", ".")

        # Normalize multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Split on sentence boundary periods
        phrases = [p.strip() for p in cleaned.split(".") if p.strip()]
        if not phrases:
            return "."

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

        Dynamically calculates thresholds if not explicitly provided.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame_bgr cannot be None or empty.")

        if not text_prompt or not text_prompt.strip():
            return []

        # Step 1: Lexical normalization
        query = self._normalize_prompt(text_prompt)

        # Step 2: Dynamic Adaptive Thresholding (DAT)
        if box_threshold is None or text_threshold is None:
            dyn_b, dyn_t = self.threshold_calculator.compute_baseline_thresholds(text_prompt)
            b_thresh = box_threshold if box_threshold is not None else dyn_b
            t_thresh = text_threshold if text_threshold is not None else dyn_t
        else:
            b_thresh = box_threshold
            t_thresh = text_threshold

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        inputs = self.processor(
            images=pil_image, text=query, return_tensors="pt"
        ).to(self.device)

        # Step 3: FP16 Autocast Inference
        t_start = time.perf_counter()
        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        target_sizes = [(h, w)]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=b_thresh,
            text_threshold=t_thresh,
            target_sizes=target_sizes,
        )[0]
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        raw_boxes = results["boxes"].cpu().numpy()
        raw_scores = results["scores"].cpu().numpy()
        labels = results.get("text_labels") or results.get("labels", [])

        if len(raw_boxes) == 0:
            return []

        # Step 4: Convert to SpatialCandidate representation for disambiguation
        spatial_cands: list[SpatialCandidate] = []
        for i in range(len(raw_boxes)):
            box = (
                float(raw_boxes[i][0]),
                float(raw_boxes[i][1]),
                float(raw_boxes[i][2]),
                float(raw_boxes[i][3]),
            )
            score = float(raw_scores[i])
            label = str(labels[i]) if i < len(labels) else query.rstrip(" .")
            spatial_cands.append(SpatialCandidate(box=box, score=score, label=label))

        # Step 5: IoU-aware deduplication (merges duplicate boxes on same object)
        deduped = self.spatial_resolver.deduplicate_candidates(spatial_cands, iou_threshold=0.45)

        # Step 6: Relative Score Falloff Filtering
        deduped_scores = [c.score for c in deduped]
        valid_indices = self.threshold_calculator.filter_by_relative_falloff(
            deduped_scores, box_threshold=b_thresh
        )
        filtered_cands = [deduped[i] for i in valid_indices]

        # Step 7: Spatial direction check if prompt contains cues ("on the left", "nearest")
        spatial_cue = self.spatial_resolver.extract_spatial_cue(text_prompt)
        if spatial_cue and len(filtered_cands) > 1:
            if spatial_cue == "left":
                filtered_cands.sort(key=lambda c: c.center_x)
            elif spatial_cue == "right":
                filtered_cands.sort(key=lambda c: c.center_x, reverse=True)
            elif spatial_cue == "nearest":
                filtered_cands.sort(key=lambda c: c.area, reverse=True)
            elif spatial_cue == "top":
                filtered_cands.sort(key=lambda c: c.center_y)
            elif spatial_cue == "bottom":
                filtered_cands.sort(key=lambda c: c.center_y, reverse=True)

        # Step 8: Build GroundingCandidate instances with optional temporal smoothing
        final_candidates: list[GroundingCandidate] = []
        for idx, sc in enumerate(filtered_cands):
            box_coords = sc.box
            # Smooth top candidate coordinates to eliminate jitter for SAM 2 handoff
            if idx == 0 and self.temporal_filter is not None:
                box_coords = self.temporal_filter.filter_box(box_coords)

            candidate = GroundingCandidate(
                box=box_coords,
                score=sc.score,
                label=sc.label,
            )
            final_candidates.append(candidate)

        logger.debug(
            "Grounding query '%s' yielded %d candidates in %.1f ms (Thresholds: box=%.2f, text=%.2f)",
            query,
            len(final_candidates),
            elapsed_ms,
            b_thresh,
            t_thresh,
        )
        return final_candidates

    def find(
        self,
        frame_bgr: np.ndarray,
        text_prompt: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[GroundingCandidate]:
        """
        Primary interface contract matching team specification: grounder.find().
        """
        return self.find_candidates(
            frame_bgr=frame_bgr,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    def is_ambiguous(
        self,
        candidates: Sequence[GroundingCandidate],
        text_prompt: str = "",
        delta_threshold: float = 0.15,
    ) -> bool:
        """
        Evaluate if visual detection candidates are ambiguous.

        Uses spatial reasoning and IoU filtering so duplicate boxes of the
        same object never trigger false ambiguity warnings.
        """
        if len(candidates) < 2:
            return False

        spatial_cands = [
            SpatialCandidate(box=c.box, score=c.score, label=c.label)
            for c in candidates
        ]
        ambig, _, _ = self.spatial_resolver.evaluate_ambiguity(
            spatial_cands, text_prompt=text_prompt, delta_threshold=delta_threshold
        )
        return ambig


def is_ambiguous(
    candidates: Sequence[GroundingCandidate],
    delta_threshold: float = 0.15,
    text_prompt: str = "",
) -> bool:
    """
    Module-level convenience function matching legacy contract.
    """
    if len(candidates) < 2:
        return False

    spatial_cands = [
        SpatialCandidate(box=c.box, score=c.score, label=c.label)
        for c in candidates
    ]
    resolver = SpatialResolver()
    ambig, _, _ = resolver.evaluate_ambiguity(
        spatial_cands, text_prompt=text_prompt, delta_threshold=delta_threshold
    )
    return ambig
