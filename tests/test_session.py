"""Tests for the session module: triggers, turn_taking, activation, compute."""

import asyncio
import time

import numpy as np
import pytest

from session.triggers import detect_triggers, filter_noise, TriggerType, ResponseHint
from session.turn_taking import TurnTakingEngine, TurnTakingConfig, Action
from session.activation import ActivationTracker, ActivationState
from session.compute import ComputeMonitor, PipelineState
from session.notes import SessionNoteProcessor
from session.timer import SessionTimer, TimerConfig


# ─── Triggers ───────────────────────────────────────────────


class TestTriggers:
    @pytest.mark.parametrize("text,expected_type", [
        ("What do you think?", TriggerType.INVITATION),
        ("What are your thoughts?", TriggerType.INVITATION),
        ("Am I being unreasonable?", TriggerType.INVITATION),
        ("Does that make sense?", TriggerType.INVITATION),
        ("Any thoughts?", TriggerType.INVITATION),
    ])
    def test_invitation_triggers(self, text, expected_type):
        r = detect_triggers(text)
        assert r.triggered
        assert r.trigger_type == expected_type
        assert r.hint == ResponseHint.SPEAK_FULL

    @pytest.mark.parametrize("text,expected_type", [
        ("Yeah idk", TriggerType.RESIGNATION),
        ("I just can't do this anymore", TriggerType.RESIGNATION),
        ("Nothing ever changes", TriggerType.RESIGNATION),
        ("It's just hard", TriggerType.RESIGNATION),
        ("What's the point", TriggerType.RESIGNATION),
    ])
    def test_resignation_triggers(self, text, expected_type):
        r = detect_triggers(text)
        assert r.triggered
        assert r.trigger_type == expected_type
        assert r.hint == ResponseHint.SPEAK_BRIEF

    @pytest.mark.parametrize("text", [
        "Is that normal?",
        "Why does this keep happening?",
        "Should I be worried?",
    ])
    def test_question_triggers(self, text):
        r = detect_triggers(text)
        assert r.triggered
        assert r.trigger_type == TriggerType.DIRECT_QUESTION

    @pytest.mark.parametrize("text", [
        "I mean...",
        "But yeah",
        "So",
    ])
    def test_trailing_off(self, text):
        r = detect_triggers(text)
        assert r.triggered
        assert r.trigger_type == TriggerType.TRAILING_OFF
        assert r.hint == ResponseHint.PLAY_CUE

    @pytest.mark.parametrize("text", [
        "My manager keeps piling on projects.",
        "I finished the report last week.",
        "The weather has been nice lately.",
    ])
    def test_no_trigger(self, text):
        r = detect_triggers(text)
        assert not r.triggered
        assert r.hint == ResponseHint.STAY_SILENT

    def test_empty_text(self):
        r = detect_triggers("")
        assert not r.triggered

    def test_invitation_priority_over_question(self):
        r = detect_triggers("What do you think about this?")
        assert r.trigger_type == TriggerType.INVITATION


class TestNoiseFilter:
    @pytest.mark.parametrize("text", [
        "um", "uh", "hmm", "ah", "oh", "like", "so", "yeah", "okay",
        "Um.", "Uh.", "Hmm.", "Right", "Well", "And", "But",
    ])
    def test_filler_only_filtered(self, text):
        assert filter_noise(text) is None

    @pytest.mark.parametrize("text", [
        "Thanks for watching", "Subscribe", "Like and subscribe",
        "[Music]", "[Applause]", "...", ".", "you",
    ])
    def test_hallucinations_filtered(self, text):
        assert filter_noise(text) is None

    @pytest.mark.parametrize("text", [
        "I feel stressed about work",
        "Yeah I don't know what to do",
        "Um, I think the problem is",
        "So like, my manager keeps",
    ])
    def test_real_speech_passes(self, text):
        assert filter_noise(text) is not None

    def test_empty_and_short(self):
        assert filter_noise("") is None
        assert filter_noise("a") is None

    def test_returns_stripped(self):
        assert filter_noise("  hello world  ") == "hello world"


# ─── Turn-Taking Engine ─────────────────────────────────────


class TestTurnTaking:
    def _engine(self, speaking_secs=25, session_age=300):
        e = TurnTakingEngine()
        e.state.session_start = time.time() - session_age
        e.update_state(user_spoke_secs=speaking_secs)
        return e

    def test_short_silence_stays_silent(self):
        e = self._engine()
        d = e.decide(1.0, 0.0)
        assert d.action == Action.STAY_SILENT

    def test_medium_silence_low_confidence_stays_silent(self):
        e = self._engine()
        d = e.decide(5.0, 0.4)
        assert d.action == Action.STAY_SILENT

    def test_medium_silence_high_confidence_plays_cue(self):
        e = self._engine()
        d = e.decide(5.0, 0.7)
        assert d.action == Action.PLAY_CUE
        assert d.cue is not None

    def test_invitation_trigger_overrides_silence(self):
        e = self._engine()
        d = e.decide(1.0, 0.0, "What do you think?")
        assert d.action == Action.SPEAK_FULL

    def test_resignation_trigger_speaks_brief(self):
        e = self._engine()
        d = e.decide(1.0, 0.0, "Yeah idk")
        assert d.action == Action.SPEAK_BRIEF

    def test_no_trigger_no_response(self):
        e = self._engine()
        d = e.decide(1.0, 0.0, "My manager piles on work.")
        assert d.action == Action.STAY_SILENT

    def test_very_long_silence_gentle_prompt(self):
        e = self._engine()
        d = e.decide(50.0, 0.7)
        assert d.action == Action.GENTLE_PROMPT

    def test_long_monologue_triggers_reflection(self):
        e = self._engine(speaking_secs=90)
        d = e.decide(7.0, 0.9)
        assert d.action == Action.SPEAK_BRIEF

    def test_emotional_content_extends_threshold(self):
        e = self._engine()
        e.update_state(emotional_content=True)
        d = e.decide(5.0, 0.7)
        assert d.action == Action.STAY_SILENT  # 4+3=7s threshold

    def test_early_session_extends_threshold(self):
        e = self._engine(session_age=30)  # just started
        d = e.decide(5.0, 0.7)
        assert d.action == Action.STAY_SILENT  # 4+2=6s threshold

    def test_backchannel_rate_limiting(self):
        e = self._engine()
        d1 = e.decide(5.0, 0.7)
        assert d1.action == Action.PLAY_CUE
        e.record_backchannel_played()
        d2 = e.decide(5.0, 0.7)
        assert d2.action == Action.STAY_SILENT  # too soon

    def test_backchannel_needs_min_speaking(self):
        e = self._engine(speaking_secs=5)  # only 5s, need 15
        d = e.decide(5.0, 0.7)
        assert d.action == Action.STAY_SILENT

    def test_cue_rotation(self):
        e = self._engine()
        d1 = e.decide(5.0, 0.7)
        cue1 = d1.cue.cue_type
        e.record_backchannel_played()
        e.state.last_backchannel_at = time.time() - 30  # reset rate limit
        d2 = e.decide(5.0, 0.7)
        cue2 = d2.cue.cue_type
        assert cue1 != cue2  # should rotate


# ─── Activation Tracker ─────────────────────────────────────


def _make_tone(freq, amplitude, duration_s=2.0, sr=16000):
    t = np.arange(int(sr * duration_s)) / sr
    noise = np.random.normal(0, 0.01, len(t))
    samples = ((amplitude * np.sin(2 * np.pi * freq * t) + noise) * 32767).astype(np.int16)
    return samples.tobytes()


class TestActivation:
    def test_baseline_building(self):
        t = ActivationTracker(baseline_chunks=3)
        for _ in range(3):
            t.process_chunk(_make_tone(150, 0.1))
        assert t._baseline_ready

    def test_baseline_not_ready_returns_default(self):
        t = ActivationTracker(baseline_chunks=5)
        state = t.process_chunk(_make_tone(150, 0.1))
        assert state.score == 0.5
        assert not t._baseline_ready

    def test_calm_vs_agitated_scores(self):
        t = ActivationTracker(baseline_chunks=3)
        for _ in range(3):
            t.process_chunk(_make_tone(150, 0.1, duration_s=3.0))

        calm_features = t.extract_features(_make_tone(150, 0.1, duration_s=3.0))
        agitated_features = t.extract_features(_make_tone(300, 0.4, duration_s=3.0))
        assert agitated_features.rms > calm_features.rms
        assert agitated_features.f0_hz > calm_features.f0_hz

    def test_trajectory_tracks_direction(self):
        t = ActivationTracker(baseline_chunks=2)
        t.process_chunk(_make_tone(150, 0.1))
        t.process_chunk(_make_tone(150, 0.1))

        t.process_chunk(_make_tone(200, 0.2))
        rising = t.process_chunk(_make_tone(250, 0.3))
        assert rising.trajectory > 0

    def test_feature_extraction(self):
        t = ActivationTracker()
        f = t.extract_features(_make_tone(200, 0.2))
        assert f.rms > 0
        assert f.f0_hz > 0
        assert f.zcr > 0

    def test_empty_audio(self):
        t = ActivationTracker()
        f = t.extract_features(b"")
        assert f.rms == 0.0


# ─── Compute Monitor ────────────────────────────────────────


# ─── Session Timer ───────────────────────────────────────────


class TestSessionTimer:
    def test_initial_state(self):
        t = SessionTimer()
        s = t.tick()
        assert s.elapsed_mins < 1
        assert not s.is_extended

    def test_no_checkin_before_threshold(self):
        t = SessionTimer(TimerConfig(gentle_checkin_mins=20))
        assert not t.should_checkin()

    def test_checkin_after_threshold(self):
        t = SessionTimer(TimerConfig(gentle_checkin_mins=5))
        t.state.started_at = time.time() - 360  # 6 minutes ago
        assert t.should_checkin()

    def test_checkin_interval_respected(self):
        t = SessionTimer(TimerConfig(gentle_checkin_mins=5, checkin_interval_mins=15))
        t.state.started_at = time.time() - 360
        assert t.should_checkin()
        t.record_checkin()
        assert not t.should_checkin()  # just checked in

    def test_extended_session(self):
        t = SessionTimer(TimerConfig(extended_session_mins=30))
        t.state.started_at = time.time() - 2400  # 40 minutes ago
        t.tick()
        assert t.state.is_extended

    def test_elapsed_display_minutes(self):
        t = SessionTimer()
        t.state.elapsed_mins = 15
        assert t.state.elapsed_display == "15m"

    def test_elapsed_display_hours(self):
        t = SessionTimer()
        t.state.elapsed_mins = 75
        assert t.state.elapsed_display == "1h 15m"

    def test_checkin_messages_vary(self):
        t = SessionTimer(TimerConfig(gentle_checkin_mins=5))
        t.state.started_at = time.time() - 600
        t.tick()
        msg1 = t.checkin_message()
        assert "reflecting" in msg1
        t.record_checkin()
        msg2 = t.checkin_message()
        assert "No rush" in msg2

    def test_extended_checkin_message(self):
        t = SessionTimer(TimerConfig(gentle_checkin_mins=5, extended_session_mins=30))
        t.state.started_at = time.time() - 2400
        t.tick()
        t.record_checkin()
        msg = t.checkin_message()
        assert "pause" in msg


# ─── Session Notes (closing ritual) ─────────────────────────


class TestSessionNotes:
    @pytest.mark.asyncio
    async def test_finalize_returns_none_without_summary(self):
        import tempfile
        d = tempfile.mkdtemp()
        p = SessionNoteProcessor(session_dir=d)
        result = await p.finalize()
        assert result is None

    @pytest.mark.asyncio
    async def test_export_to_journal(self):
        import tempfile
        session_dir = tempfile.mkdtemp()
        journal_dir = tempfile.mkdtemp()
        p = SessionNoteProcessor(session_dir=session_dir)
        p.running_summary = "User discussed work stress and boundaries."
        p.active_themes = ["work stress", "boundaries"]
        p.chunk_index = 5

        path = await p.export_to_journal(journal_dir)
        assert path is not None
        from pathlib import Path
        content = Path(path).read_text()
        assert "work stress" in content
        assert "boundaries" in content
        assert "5 exchanges" in content

    @pytest.mark.asyncio
    async def test_export_empty_session_returns_none(self):
        import tempfile
        p = SessionNoteProcessor(session_dir=tempfile.mkdtemp())
        result = await p.export_to_journal(tempfile.mkdtemp())
        assert result is None

    def test_session_dir_created(self):
        import tempfile
        d = tempfile.mkdtemp() + "/new-session"
        p = SessionNoteProcessor(session_dir=d)
        from pathlib import Path
        assert Path(d).exists()
        assert (Path(d) / "meta.json").exists()
        assert (Path(d) / "wiki").is_dir()


# ─── Compute Monitor ────────────────────────────────────────


class TestComputeMonitor:
    def test_initial_state(self):
        m = ComputeMonitor()
        assert m.state.pipeline == PipelineState.IDLE
        assert not m.should_cancel_llm()

    def test_vad_start_gates_llm(self):
        m = ComputeMonitor()
        m.on_vad_start()
        assert m.state.user_speaking
        assert m.should_cancel_llm()
        assert not m._llm_gate.is_set()

    def test_vad_stop_ungates_llm(self):
        m = ComputeMonitor()
        m.on_vad_start()
        m.on_vad_stop()
        assert not m.state.user_speaking
        assert not m.should_cancel_llm()
        assert m._llm_gate.is_set()

    def test_background_llm_blocked_during_stt(self):
        m = ComputeMonitor()
        m.on_vad_stop()
        m.on_stt_start()
        assert not m.can_run_background_llm()
        m.on_stt_done()
        assert m.can_run_background_llm()

    def test_interrupt_during_llm(self):
        m = ComputeMonitor()
        m.on_llm_start()
        assert m.state.pipeline == PipelineState.THINKING
        m.on_vad_start()
        assert m.should_cancel_llm()
        assert m.should_cancel_tts()

    @pytest.mark.asyncio
    async def test_gate_blocks_and_releases(self):
        m = ComputeMonitor()
        m.on_vad_start()

        async def wait():
            await m.gate_llm()
            return True

        task = asyncio.create_task(wait())
        await asyncio.sleep(0.05)
        assert not task.done()
        m.on_vad_stop()
        await asyncio.sleep(0.05)
        assert task.done()
        assert task.result() is True
