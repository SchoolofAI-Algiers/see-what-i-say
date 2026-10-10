from stark_vision.contracts import CommandType, EffectMode
from stark_vision.speech.command_parser import CommandParser

parser = CommandParser()


def test_follow_pronoun_keeps_current_target():
    assert parser.parse("follow her").type is CommandType.FOLLOW
    assert parser.parse("track him").type is CommandType.FOLLOW


def test_select_target_adds_article():
    ev = parser.parse("STARK, segment guy wearing red")
    assert ev.type is CommandType.SELECT_TARGET
    assert ev.prompt == "the guy wearing red"


def test_effects_and_stop_variants():
    assert parser.parse("blur him").effect.mode is EffectMode.BLUR
    assert parser.parse("make her blue").effect.mode is EffectMode.RECOLOR
    for text in ("stop tracking", "reset", "clear everything", "clear all"):
        assert parser.parse(text).type is CommandType.STOP_TRACKING


def test_unknown():
    assert parser.parse("what is he doing").type is CommandType.UNKNOWN
