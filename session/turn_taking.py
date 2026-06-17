"""
Turn-taking engine for RestReflect.

Combines multiple signals to decide: stay silent, play a backchannel
cue, or generate a spoken response. Default is silence — this is the
user's space.

Signals:
    - NLP triggers (fast regex, <1ms)
    - Silence duration (from VAD)
    - Smart Turn confidence (from Pipecat, 0.0-1.0)
    - LLM assess_moment output (from SessionNoteProcessor)
    - Conversation state (time speaking, time since last response, etc.)
"""

import time
from dataclasses import dataclass, field
from enum import Enum

from session.cue_rotation import CUE_ROTATION, cue_for_index
from session.triggers import TriggerResult, TriggerType, ResponseHint, detect_triggers


class Action(Enum):
    STAY_SILENT = "stay_silent"
    PLAY_CUE = "play_cue"
    SPEAK_BRIEF = "speak_brief"
    SPEAK_FULL = "speak_full"
    GENTLE_PROMPT = "gentle_prompt"


@dataclass
class CueSelection:
    cue_type: str = "mhmm"


@dataclass
class TurnDecision:
    action: Action
    reason: str
    cue: CueSelection | None = None
    confidence: float = 0.0


def plan_cue_broadcast(
    decision: "TurnDecision",
    *,
    trigger_fired: bool = False,
) -> str | None:
    """Decide *which* backchannel cue (if any) a live cue path should broadcast.

    The single gate the live sidecar (``pipecat_server.py``'s ``Broadcaster``)
    consults after :meth:`TurnTakingEngine.decide`: returns the ``cue_type`` to
    play, or ``None`` to stay silent. Pure, dependency-free, file-path-loadable
    in tests like its sibling seams.

    A cue is broadcast iff **all** hold:

      - ``decision.action`` is :attr:`Action.PLAY_CUE` (the engine chose the
        backchannel tier), and
      - an NLP trigger did *not* fire (``trigger_fired`` is ``False``) — a
        trigger means the user said something that warrants a real response, so
        a continuer "mhmm" would step on it, and
      - the decision actually carries a cue (``decision.cue`` is not ``None``).

    Crucially the returned cue is the engine's **rotated** cue
    (``decision.cue.cue_type``), advanced through the shared ``CUE_ROTATION``
    (``session/cue_rotation.py``) by :meth:`TurnTakingEngine._next_cue`. The
    live path must NOT re-pick a random cue from a private list: that both
    drops the rotation (the agent repeats sounds) and risks playing a cue key
    outside the shared rotation. This function is what keeps the live broadcast
    indexed to the one source of truth.
    """
    if trigger_fired:
        return None
    if decision.action is not Action.PLAY_CUE:
        return None
    if decision.cue is None:
        return None
    return decision.cue.cue_type


@dataclass
class ConversationState:
    session_start: float = field(default_factory=time.time)
    user_speaking_total_secs: float = 0.0
    last_system_spoke_at: float | None = None
    last_backchannel_at: float | None = None
    chunks_since_last_response: int = 0
    emotional_content_recent: bool = False
    user_crying: bool = False

    @property
    def session_age_secs(self) -> float:
        return time.time() - self.session_start

    @property
    def secs_since_system_spoke(self) -> float | None:
        if self.last_system_spoke_at is None:
            return None
        return time.time() - self.last_system_spoke_at

    @property
    def secs_since_backchannel(self) -> float | None:
        if self.last_backchannel_at is None:
            return None
        return time.time() - self.last_backchannel_at


@dataclass
class TurnTakingConfig:
    # Silence thresholds (seconds)
    # ``silence_backchannel_min`` is the lower edge of the silence-driven
    # turn-end cue window: a ``PLAY_CUE`` fires once trailing silence reaches
    # this. It must sit at/above ``backchannel_timing.py``'s ``max_pause_secs``
    # (2.0s, where the mid-speech backchannel window closes) so the two cue
    # paths partition the silence axis with no overlap — no gap fires both a
    # mid-speech "mhmm" and a turn-end cue. That inequality is pinned by
    # ``tests/unit/test_cue_window_partition_invariant.py`` (iter-180) against
    # the sibling module, not a hardcoded ``2.0``.
    silence_backchannel_min: float = 4.0
    silence_response_min: float = 6.0
    silence_gentle_prompt: float = 45.0

    # Smart Turn confidence thresholds
    smart_turn_backchannel_min: float = 0.6
    smart_turn_response_min: float = 0.85

    # Adaptive adjustments
    emotional_extension_secs: float = 3.0
    crying_extension_secs: float = 10.0
    early_session_extension_secs: float = 2.0
    early_session_window_secs: float = 120.0

    # Backchannel limits. These two scalars are deliberately mirrored by
    # ``backchannel_timing.py``'s ``BackchannelTimingConfig`` (same 15.0 / 20.0)
    # so the silence-driven turn-end cue path and the mid-speech backchannel
    # path obey one shared cadence — the agent feels like one speaker, not two
    # subsystems with divergent tempos. That cross-module equality is pinned by
    # ``tests/unit/test_backchannel_rate_limit_mirror_invariant.py`` (iter-181)
    # against the sibling config, not a hardcoded literal.
    min_speaking_before_first_cue_secs: float = 15.0
    min_between_cues_secs: float = 20.0

    # Speaking thresholds
    long_monologue_secs: float = 60.0


# ``CUE_ROTATION`` now lives in ``session/cue_rotation.py`` (the single source
# of truth shared with the mid-speech ``BackchannelMonitor``); it is re-exported
# here for backward compatibility with existing importers.


class TurnTakingEngine:
    def __init__(self, config: TurnTakingConfig | None = None):
        self.config = config or TurnTakingConfig()
        self.state = ConversationState()
        self._cue_index = 0
        self._last_nlp_trigger: TriggerResult | None = None
        self._last_llm_assessment: dict | None = None

    def update_state(
        self,
        user_spoke_secs: float = 0.0,
        emotional_content: bool | None = None,
        user_crying: bool | None = None,
    ):
        self.state.user_speaking_total_secs += user_spoke_secs
        self.state.chunks_since_last_response += 1 if user_spoke_secs > 0 else 0
        if emotional_content is not None:
            self.state.emotional_content_recent = emotional_content
        if user_crying is not None:
            self.state.user_crying = user_crying

    def record_nlp_trigger(self, trigger: TriggerResult):
        self._last_nlp_trigger = trigger

    def record_llm_assessment(self, assessment: dict):
        self._last_llm_assessment = assessment

    def record_system_spoke(self):
        self.state.last_system_spoke_at = time.time()
        self.state.chunks_since_last_response = 0

    def record_backchannel_played(self):
        self.state.last_backchannel_at = time.time()

    def decide(
        self,
        silence_duration_secs: float,
        smart_turn_confidence: float = 0.0,
        transcript_chunk: str | None = None,
    ) -> TurnDecision:
        cfg = self.config

        if transcript_chunk:
            trigger = detect_triggers(transcript_chunk)
            self.record_nlp_trigger(trigger)
        else:
            trigger = self._last_nlp_trigger or TriggerResult(triggered=False)

        thresholds = self._compute_thresholds()
        bc_min = thresholds["backchannel_min"]
        resp_min = thresholds["response_min"]
        gentle_min = cfg.silence_gentle_prompt

        # --- NLP trigger override (fastest signal) ---
        if trigger.triggered:
            if trigger.hint == ResponseHint.SPEAK_FULL:
                return TurnDecision(
                    action=Action.SPEAK_FULL,
                    reason=f"NLP trigger: {trigger.trigger_type.value} — \"{trigger.pattern_matched}\"",
                    confidence=trigger.confidence,
                )
            if trigger.hint == ResponseHint.SPEAK_BRIEF:
                return TurnDecision(
                    action=Action.SPEAK_BRIEF,
                    reason=f"NLP trigger: {trigger.trigger_type.value}",
                    confidence=trigger.confidence,
                )

        # --- LLM assessment override ---
        if self._last_llm_assessment:
            llm_action = self._last_llm_assessment.get("action", "stay_silent")
            if llm_action in ("speak_brief", "speak_full"):
                return TurnDecision(
                    action=Action.SPEAK_BRIEF if llm_action == "speak_brief" else Action.SPEAK_FULL,
                    reason=f"LLM assessment: {self._last_llm_assessment.get('reason', '')}",
                    confidence=0.7,
                )

        # --- Tier 0: Too soon ---
        if silence_duration_secs < bc_min:
            return TurnDecision(
                action=Action.STAY_SILENT,
                reason=f"Silence {silence_duration_secs:.1f}s < {bc_min:.1f}s threshold",
            )

        # --- Tier 1: Smart Turn not confident ---
        if smart_turn_confidence < cfg.smart_turn_backchannel_min:
            return TurnDecision(
                action=Action.STAY_SILENT,
                reason=f"Smart Turn confidence {smart_turn_confidence:.2f} < {cfg.smart_turn_backchannel_min:.2f}",
            )

        # --- Tier 2: Backchannel window ---
        if silence_duration_secs < resp_min:
            if self._backchannel_appropriate():
                cue = self._next_cue()
                return TurnDecision(
                    action=Action.PLAY_CUE,
                    reason=f"Backchannel window ({silence_duration_secs:.1f}s silence, ST={smart_turn_confidence:.2f})",
                    cue=cue,
                    confidence=smart_turn_confidence,
                )
            return TurnDecision(
                action=Action.STAY_SILENT,
                reason="Backchannel window but rate-limited or too early",
            )

        # --- Tier 3: LLM response window ---
        if smart_turn_confidence >= cfg.smart_turn_response_min:
            if trigger.triggered and trigger.trigger_type in (TriggerType.INVITATION, TriggerType.DIRECT_QUESTION):
                return TurnDecision(
                    action=Action.SPEAK_FULL,
                    reason=f"Extended silence ({silence_duration_secs:.1f}s) + high ST ({smart_turn_confidence:.2f}) + {trigger.trigger_type.value}",
                    confidence=smart_turn_confidence,
                )
            if self.state.user_speaking_total_secs > cfg.long_monologue_secs:
                return TurnDecision(
                    action=Action.SPEAK_BRIEF,
                    reason=f"Extended silence after {self.state.user_speaking_total_secs:.0f}s monologue — offer a reflection",
                    confidence=smart_turn_confidence,
                )
            return TurnDecision(
                action=Action.STAY_SILENT,
                reason=f"Extended silence but no invitation and monologue < {cfg.long_monologue_secs}s — default to silence",
            )

        # --- Tier 4: Very long silence ---
        if silence_duration_secs >= gentle_min:
            return TurnDecision(
                action=Action.GENTLE_PROMPT,
                reason=f"Very long silence ({silence_duration_secs:.0f}s) — gentle prompt",
                confidence=0.5,
            )

        return TurnDecision(
            action=Action.STAY_SILENT,
            reason="No signals warrant a response",
        )

    def _compute_thresholds(self) -> dict:
        cfg = self.config
        bc_min = cfg.silence_backchannel_min
        resp_min = cfg.silence_response_min

        if self.state.user_crying:
            bc_min += cfg.crying_extension_secs
            resp_min += cfg.crying_extension_secs
        elif self.state.emotional_content_recent:
            bc_min += cfg.emotional_extension_secs
            resp_min += cfg.emotional_extension_secs

        if self.state.session_age_secs < cfg.early_session_window_secs:
            bc_min += cfg.early_session_extension_secs
            resp_min += cfg.early_session_extension_secs

        return {"backchannel_min": bc_min, "response_min": resp_min}

    def _backchannel_appropriate(self) -> bool:
        if self.state.user_speaking_total_secs < self.config.min_speaking_before_first_cue_secs:
            return False
        since = self.state.secs_since_backchannel
        if since is not None and since < self.config.min_between_cues_secs:
            return False
        return True

    def _next_cue(self) -> CueSelection:
        cue = cue_for_index(self._cue_index)
        self._cue_index += 1
        return CueSelection(cue_type=cue)
