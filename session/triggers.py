"""
NLP trigger detector for the turn-taking engine.

Fast pattern matching on transcript text to detect when the user
is inviting a response, expressing resignation, asking a question,
or at an emotional peak. No LLM needed — runs in <1ms on regex.

Each trigger returns a signal that feeds into the turn-taking engine
alongside other inputs (silence duration, Smart Turn confidence,
LLM background assessment).
"""

import re
from dataclasses import dataclass
from enum import Enum

# Whisper hallucination patterns (common artifacts on silence/noise)
_HALLUCINATION_PATTERNS = [
    re.compile(r"^(thanks for watching|thank you for watching)", re.IGNORECASE),
    re.compile(r"^(subscribe|like and subscribe)", re.IGNORECASE),
    re.compile(r"^(music|applause|\[music\]|\[applause\])\s*$", re.IGNORECASE),
    re.compile(r"^\.+$"),
    re.compile(r"^(you|\.)\s*$", re.IGNORECASE),
]

# Filler-only utterances (not meaningful speech)
_FILLER_ONLY = re.compile(
    r"^(um+|uh+|hmm+|hm+|ah+|oh+|er+|like|so|yeah|okay|ok|right|well|and|but)\s*\.?\s*$",
    re.IGNORECASE,
)


def filter_noise(text: str) -> str | None:
    """Filter out noise, hallucinations, and filler-only utterances.

    Returns the cleaned text, or None if the entire chunk should be
    discarded (hallucination, pure filler, or too short to be meaningful).
    """
    text = text.strip()

    if not text or len(text) < 2:
        return None

    for pattern in _HALLUCINATION_PATTERNS:
        if pattern.match(text):
            return None

    if _FILLER_ONLY.match(text):
        return None

    return text


class TriggerType(Enum):
    DIRECT_QUESTION = "direct_question"
    INVITATION = "invitation"
    RESIGNATION = "resignation"
    EMOTIONAL_PEAK = "emotional_peak"
    TRAILING_OFF = "trailing_off"


class ResponseHint(Enum):
    SPEAK_FULL = "speak_full"
    SPEAK_BRIEF = "speak_brief"
    PLAY_CUE = "play_cue"
    STAY_SILENT = "stay_silent"


@dataclass
class TriggerResult:
    triggered: bool
    trigger_type: TriggerType | None = None
    hint: ResponseHint = ResponseHint.STAY_SILENT
    pattern_matched: str | None = None
    confidence: float = 0.0


# Direct invitations — user explicitly asks for a response
_INVITATION_PATTERNS = [
    (r"\bwhat do you think\b", 0.95),
    (r"\bwhat are your thoughts\b", 0.95),
    (r"\bwhat would you say\b", 0.9),
    (r"\bany (ideas|thoughts|suggestions)\b", 0.85),
    (r"\bdo you have any (thoughts|input|advice)\b", 0.9),
    (r"\bcan you help me (think|figure|understand|see)\b", 0.85),
    (r"\bwhat should i do\b", 0.9),
    (r"\bam i (being |making |)?(unreasonable|crazy|wrong|right|overreacting|overthinking)\b", 0.9),
    (r"\bdoes that make (any )?sense\b", 0.85),
    (r"\bi('d| would) (love|like|appreciate) (to hear |)(your |)thoughts\b", 0.9),
]

# Resignation / surrender — user has run out of steam
_RESIGNATION_PATTERNS = [
    (r"\bi don'?t know\s*(anymore|what to do|what to think)?\s*$", 0.85),
    (r"\byeah i(dk| don'?t know)\b", 0.9),
    (r"\bit('?s| is) (just |so |really )?(hard|tough|difficult|exhausting|draining)\s*$", 0.85),
    (r"\bi('?m| am) (just |so |really )?(tired|exhausted|done|over it|burnt out)\b", 0.85),
    (r"\bi (just |)(can'?t|cannot) (do this|take it|handle|deal with)\b", 0.9),
    (r"\bi give up\b", 0.9),
    (r"\bwhatever\s*$", 0.7),
    (r"\bit is what it is\b", 0.8),
    (r"\bnothing (ever |)(works|changes|helps|matters)\b", 0.85),
    (r"\bwhat'?s the point\b", 0.9),
]

# Trailing off — incomplete thought, fading energy
_TRAILING_PATTERNS = [
    (r"\.\.\.\s*$", 0.6),
    (r"\bi (don'?t know|guess|mean)\s*\.?\s*$", 0.7),
    (r"\bbut (yeah|anyway|whatever)\s*\.?\s*$", 0.75),
    (r"\bso\s*\.?\s*$", 0.6),
    (r"\byeah\s*\.?\s*$", 0.5),
]

# Direct questions — ends with question mark or interrogative structure
_QUESTION_PATTERNS = [
    (r"\?\s*$", 0.7),
    (r"\b(is|are|was|were|do|does|did|can|could|should|would|will) (that|this|it|i|we|they)\b.*\?\s*$", 0.85),
    (r"\b(why|how|what|when|where|who)\b.*\?\s*$", 0.8),
]

# Emotional intensity — repeated words, strong language
_EMOTIONAL_PATTERNS = [
    (r"\b(never|always|nobody|everybody|nothing|everything)\b.*\b(never|always|nobody|everybody|nothing|everything)\b", 0.8),
    (r"\bi (hate|love|can'?t stand|can'?t believe)\b", 0.7),
    (r"\b(so |really |very |extremely ){2,}", 0.75),
    (r"(!){2,}", 0.7),
]


def _check_patterns(
    text: str,
    patterns: list[tuple[str, float]],
    trigger_type: TriggerType,
    hint: ResponseHint,
) -> TriggerResult | None:
    for pattern, confidence in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return TriggerResult(
                triggered=True,
                trigger_type=trigger_type,
                hint=hint,
                pattern_matched=pattern,
                confidence=confidence,
            )
    return None


def detect_triggers(text: str) -> TriggerResult:
    """Detect NLP triggers in a transcript chunk.

    Returns the highest-priority trigger found, or a no-trigger result.
    Priority: invitation > resignation > question > emotional > trailing.
    """
    text = text.strip()
    if not text:
        return TriggerResult(triggered=False)

    result = _check_patterns(text, _INVITATION_PATTERNS, TriggerType.INVITATION, ResponseHint.SPEAK_FULL)
    if result:
        return result

    result = _check_patterns(text, _RESIGNATION_PATTERNS, TriggerType.RESIGNATION, ResponseHint.SPEAK_BRIEF)
    if result:
        return result

    result = _check_patterns(text, _QUESTION_PATTERNS, TriggerType.DIRECT_QUESTION, ResponseHint.SPEAK_BRIEF)
    if result:
        return result

    result = _check_patterns(text, _EMOTIONAL_PATTERNS, TriggerType.EMOTIONAL_PEAK, ResponseHint.PLAY_CUE)
    if result:
        return result

    result = _check_patterns(text, _TRAILING_PATTERNS, TriggerType.TRAILING_OFF, ResponseHint.PLAY_CUE)
    if result:
        return result

    return TriggerResult(triggered=False)
