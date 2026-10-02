from enum import Enum
from dataclasses import dataclass
import re
from stt import SpeechToText, recording_fnct


class Action(Enum):
    SELECT_TARGET = 1
    SET_EFFECT = 2
    FOLLOW = 3
    STOP_TRACKING = 4
    RESET = 5
    # resolve_ambiguity

class Effect(Enum):
    HIGHLIGHT = 1
    RECOLOR = 2
    BLUR = 3
    GLOW = 4
    INVISIBLE = 5

@dataclass(frozen=True)
class EffectSpec():
    type : Effect
    color : str | None = None

@dataclass(frozen=True)
class Command:
    action : Action
    referring_expression : str | None
    effect : EffectSpec | None

class CommandParser:

    def parse(self, sentence):
        if "stop tracking" in sentence :
            command = Command(action=Action.STOP_TRACKING,referring_expression=None,effect=None)
            return command
        elif "reset" in sentence:
            command = Command(action=Action.RESET,referring_expression=None,effect=None)
            return command
        else:
            tracking_verbs = ["track","follow"]
            for verb in tracking_verbs:
                pattern = rf"{re.escape(verb)}\s+(.*)"
                searched_pattern = re.search(pattern, sentence, re.IGNORECASE)
                if searched_pattern is not None :
                    act = Action.FOLLOW
                    command = Command(action= act, referring_expression=searched_pattern.group(1), effect=None)
                    return command
            
            selection_verbs = [ "select", "choose", "segment"]
            for verb in selection_verbs : 
                pattern = rf"{re.escape(verb)}\s+(.*)"
                searched_pattern = re.search(pattern, sentence, re.IGNORECASE)
                if searched_pattern is not None :
                    act = Action.SELECT_TARGET
                    target = searched_pattern.group(1)
                    command = Command(action= act, referring_expression=target, effect=None)
                    return command

            effect_verbs_highlight =["highlight"]
            for verb in effect_verbs_highlight:
                pattern = rf"{re.escape(verb)}\s+(.*)"
                searched_pattern = re.search(pattern, sentence, re.IGNORECASE)
                if searched_pattern is not None :
                    act = Action.SET_EFFECT
                    target = searched_pattern.group(1)
                    effect = EffectSpec(type=Effect.HIGHLIGHT)
                    command = Command(action= act, referring_expression=target, effect=effect)
                    return command
                
            if "blur" in sentence:
                                pattern = rf"{re.escape("blur")}\s+(.*)"
                                searched_pattern = re.search(pattern, sentence, re.IGNORECASE)
                                if searched_pattern is not None  : 
                                    target = searched_pattern.group(1)
                                    act = Action.SET_EFFECT
                                    effect = EffectSpec(type=Effect.BLUR)
                                    command = Command(action= act, referring_expression=target, effect=effect)
                                    return command
            
            if "make" in sentence :
                pattern = rf"make\s+(it|him|her|the object)\s+(.*)"
                searched_pattern = re.search(pattern, sentence, re.IGNORECASE)

                if searched_pattern is not None and searched_pattern.group(2) == "glow":
                    target = searched_pattern.group(1)
                    act = Action.SET_EFFECT
                    effect = EffectSpec(type=Effect.GLOW)
                    command = Command(action= act, referring_expression=target, effect=effect)
                    return command

                elif searched_pattern is not None and searched_pattern.group(2) == "invisible" :
                    target = searched_pattern.group(1)
                    act = Action.SET_EFFECT
                    effect = EffectSpec(type=Effect.INVISIBLE)
                    command = Command(action= act, referring_expression=target, effect=effect)
                    return command 
                   
            effect_verbs_color=["color","recolor","make"]
            for verb in effect_verbs_color:
                pattern = rf"{re.escape(verb)}\s+(it|him|her|the color)\s+(.*)"
                searched_pattern = re.search(pattern, sentence, re.IGNORECASE)
                if searched_pattern is not None :
                    act = Action.SET_EFFECT
                    color = searched_pattern.group(2)
                    target = searched_pattern.group(1)
                    effect = EffectSpec(type=Effect.RECOLOR, color=color)
                    command = Command(action= act, referring_expression=target, effect=effect)
                    return command

record = recording_fnct()
stt = SpeechToText()
non_normalized_sentence = stt.transcription(record)
print("Transcribed sentence:", non_normalized_sentence)
lower_sentence = non_normalized_sentence.lower()
sentence = lower_sentence.strip()
parser = CommandParser()
print(parser.parse(sentence))


# Tests = [
# "stop tracking",
# "reset",
# "follow her",
# "segment the man in red",
# "highlight the woman on the left",
# "make it blue",
# "blur it",
# "make it glow",
# "make it invisible",
# "make him glow",
# "make her invisible"
# ]
# for non_normalized_sentence in Tests : 
#     sentence = non_normalized_sentence.lower()
#     parser = CommandParser()
#     print(parser.parse(sentence))

# # Results: 
# # Command(action=<Action.STOP_TRACKING: 4>, referring_expression=None, effect=None)
# # Command(action=<Action.RESET: 5>, referring_expression=None, effect=None)
# # Command(action=<Action.FOLLOW: 3>, referring_expression='her', effect=None)
# # Command(action=<Action.SELECT_TARGET: 1>, referring_expression='the man in red', effect=None)
# # Command(action=<Action.SET_EFFECT: 2>, referring_expression='the woman on the left', effect=EffectSpec(type=<Effect.HIGHLIGHT: 1>, color=None))
# # Command(action=<Action.SET_EFFECT: 2>, referring_expression='it', effect=EffectSpec(type=<Effect.RECOLOR: 2>, color='blue'))
# # Command(action=<Action.SET_EFFECT: 2>, referring_expression='it', effect=EffectSpec(type=<Effect.BLUR: 3>, color=None))
# # Command(action=<Action.SET_EFFECT: 2>, referring_expression='it', effect=EffectSpec(type=<Effect.GLOW: 4>, color=None))
# # Command(action=<Action.SET_EFFECT: 2>, referring_expression='it', effect=EffectSpec(type=<Effect.INVISIBLE: 5>, color=None))
# # Command(action=<Action.SET_EFFECT: 2>, referring_expression='him', effect=EffectSpec(type=<Effect.GLOW: 4>, color=None))
# # Command(action=<Action.SET_EFFECT: 2>, referring_expression='her', effect=EffectSpec(type=<Effect.INVISIBLE: 5>, color=None))