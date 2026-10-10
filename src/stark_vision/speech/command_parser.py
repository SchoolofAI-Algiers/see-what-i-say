"""Deterministic parser for the MVP voice command vocabulary."""

from __future__ import annotations

import re

from stark_vision.contracts import (
    CandidateSelector,
    CandidateSelectorKind,
    CommandEvent,
    CommandType,
    EffectMode,
    EffectSpec,
)


_COLORS_RGB = {
    "red": (255, 0, 0),
    "orange": (255, 128, 0),
    "yellow": (255, 220, 0),
    "green": (0, 200, 0),
    "blue": (0, 80, 255),
    "purple": (150, 50, 220),
    "pink": (255, 80, 170),
    "cyan": (0, 220, 220),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_PRONOUNS = {"him", "her", "it", "them", "the target"}
_PREFIX = re.compile(r"^(?:stark\s*[,.:]?\s*)", re.IGNORECASE)


class CommandParser:
    """Parse spoken text into application-level events, without model I/O."""

    def parse(self, transcript: str) -> CommandEvent:
        original = transcript
        text = _PREFIX.sub("", transcript.strip().lower()).strip(" .!?\t")

        if re.fullmatch(r"(?:stop tracking|reset|clear target|clear the target)", text):
            return CommandEvent(CommandType.STOP_TRACKING, original)

        side_match = re.fullmatch(r"(?:choose |select )?(?:the )?one on the (left|right)", text)
        if side_match:
            selector = CandidateSelector(CandidateSelectorKind.SIDE, side_match.group(1))
            return CommandEvent(CommandType.RESOLVE_CANDIDATE, original, selector=selector)

        rank_match = re.fullmatch(r"(?:number|candidate)\s+(one|two|three|four|five|[1-5])", text)
        if rank_match:
            value = rank_match.group(1)
            rank = _NUMBER_WORDS[value] if value in _NUMBER_WORDS else int(value)
            selector = CandidateSelector(CandidateSelectorKind.RANK, rank)
            return CommandEvent(CommandType.RESOLVE_CANDIDATE, original, selector=selector)

        recolor = re.fullmatch(
            r"(?:make|color|colour|recolor|recolour)\s+(?:him|her|it|them|the target)\s+(?:more\s+)?([a-z]+)",
            text,
        )
        if recolor and recolor.group(1) in _COLORS_RGB:
            effect = EffectSpec(EffectMode.RECOLOR, _COLORS_RGB[recolor.group(1)])
            return CommandEvent(CommandType.SET_EFFECT, original, effect=effect)

        effect_match = re.fullmatch(
            r"(blur|isolate|highlight)\s+(?:him|her|it|them|the target)", text
        )
        if effect_match:
            effect = EffectSpec(EffectMode(effect_match.group(1)))
            return CommandEvent(CommandType.SET_EFFECT, original, effect=effect)

        selection = re.fullmatch(
            r"(?:please\s+)?(segment|select|track|follow)\s+(.+)", text
        )
        if selection:
            verb = selection.group(1)
            prompt = selection.group(2).strip(" .!?\t")

            # "follow her" / "track him" -> keep the current target, no new grounding
            if verb in ("follow", "track") and prompt in _PRONOUNS:
                return CommandEvent(CommandType.FOLLOW, original)

            if prompt and not prompt.startswith(("the ", "a ", "an ")):
                prompt = "the " + prompt
            if prompt:
                return CommandEvent(CommandType.SELECT_TARGET, original, prompt=prompt)

        return CommandEvent(CommandType.UNKNOWN, original)
