"""
E2E pipeline integration tests.

Tests the full MindReflect pipeline using diarized clips routed through
internal loopback audio. Each test plays clips → sidecar captures via
loopback → STT → triggers → WebSocket → verifies output.

Run with:
    .venv/bin/python -m pytest tests/e2e/ -v --tb=short

Requires:
    - Voice server running (server.py)
    - Sidecar running with Loopback Audio as input (pipecat_server.py)
    - SwitchAudioSource installed (brew install switchaudio-osx)
    - Diarized clips in test-data/diarization/
"""

import asyncio
import time

import pytest


class TestSTTDirect:
    """Test STT via HTTP API (no loopback needed)."""

    def test_transcribe_clip(self, voice_server, esther_ep1):
        clips = esther_ep1.clips_with_text()
        assert len(clips) > 0
        clip = clips[0]
        wav = open(clip["path"], "rb").read()
        result = voice_server.transcribe(wav)
        assert result.get("text"), f"No transcription for {clip['clip']}"
        assert len(result["text"]) > 5

    def test_trigger_detection(self, voice_server, esther_ep1):
        clips = esther_ep1.clips_with_text()
        triggered = 0
        for clip in clips[:20]:
            wav = open(clip["path"], "rb").read()
            result = voice_server.transcribe(wav)
            if result.get("trigger"):
                triggered += 1
        assert triggered > 0, "No triggers detected in first 20 clips"

    def test_daic_transcribe(self, voice_server, daic_300):
        clips = daic_300.clips_with_text()
        assert len(clips) > 0
        clip = clips[0]
        wav = open(clip["path"], "rb").read()
        result = voice_server.transcribe(wav)
        assert result.get("text")


class TestTTS:
    """Test TTS synthesis."""

    def test_synthesize(self, voice_server):
        wav = voice_server.synthesize("Hello, how are you feeling today?")
        assert len(wav) > 1000, "TTS output too small"

    def test_synthesize_voices(self, voice_server):
        for voice in ["am_michael", "af_heart"]:
            wav = voice_server.synthesize("Test.", voice=voice)
            assert len(wav) > 500, f"TTS failed for voice {voice}"


class TestNotes:
    """Test session notes pipeline."""

    def test_process_and_themes(self, voice_server):
        voice_server.process_note("I feel overwhelmed by everything happening at work.")
        voice_server.process_note("I used to enjoy painting but haven't in months.")
        time.sleep(15)
        themes = voice_server.notes_themes()
        assert themes.get("chunks", 0) >= 2

    def test_notes_from_clips(self, voice_server, esther_ep1):
        clips = esther_ep1.clips_with_text()[:5]
        for clip in clips:
            voice_server.process_note(clip["text"])
        time.sleep(10)
        themes = voice_server.notes_themes()
        assert themes.get("chunks", 0) >= 5


class TestLoopbackPipeline:
    """Full E2E: play clips → loopback → sidecar → WebSocket → verify."""

    @pytest.mark.asyncio
    async def test_single_clip_roundtrip(self, voice_server, loopback, esther_ep1, sidecar_collector):
        clips = esther_ep1.clips_with_text()
        clip = clips[5]  # pick one with decent text

        # Play clip through loopback
        loopback.play(clip["path"])
        time.sleep(1)

        # Collect from sidecar
        messages = await sidecar_collector.collect(duration=10)
        transcripts = sidecar_collector.transcripts

        assert len(transcripts) > 0, "No transcripts received from sidecar"
        combined = " ".join(t["text"] for t in transcripts)
        assert len(combined) > 10, f"Transcript too short: {combined}"

    @pytest.mark.asyncio
    async def test_multi_clip_conversation(self, voice_server, loopback, esther_ep1, sidecar_collector):
        clips = esther_ep1.clips_with_text()[:10]

        # Play clips sequentially with pauses
        for clip in clips:
            loopback.play(clip["path"])
            time.sleep(0.5)

        # Collect all transcripts
        messages = await sidecar_collector.collect(duration=30)
        transcripts = sidecar_collector.transcripts

        assert len(transcripts) >= 3, f"Expected >=3 transcripts, got {len(transcripts)}"

    @pytest.mark.asyncio
    async def test_trigger_fires_on_question(self, voice_server, loopback, esther_ep1, sidecar_collector):
        # Find a clip that's a known trigger (question)
        clips = esther_ep1.clips_with_text()
        question_clip = None
        for c in clips:
            text = c["text"].lower()
            if "?" in text and len(c["text"]) > 20:
                question_clip = c
                break

        if not question_clip:
            pytest.skip("No question clip found")

        loopback.play(question_clip["path"])
        time.sleep(2)

        messages = await sidecar_collector.collect(duration=15)
        triggers = sidecar_collector.triggers

        assert len(triggers) > 0, f"No trigger from question: {question_clip['text'][:60]}"

    @pytest.mark.asyncio
    async def test_notes_accumulate_during_playback(self, voice_server, loopback, esther_ep1, sidecar_collector):
        initial = voice_server.notes_themes().get("chunks", 0)

        clips = esther_ep1.clips_with_text()[:5]
        for clip in clips:
            loopback.play(clip["path"])
            time.sleep(0.5)

        # Wait for sidecar → notes pipeline
        await sidecar_collector.collect(duration=20)
        time.sleep(10)

        final = voice_server.notes_themes().get("chunks", 0)
        assert final > initial, f"Notes didn't accumulate: {initial} → {final}"


class TestClipQueue:
    """Test clip queue data integrity."""

    def test_esther_clips_have_transcripts(self, esther_ep1):
        clips = esther_ep1.clips_with_text()
        total = len(esther_ep1.clips_for())
        ratio = len(clips) / total if total > 0 else 0
        assert ratio > 0.7, f"Only {ratio:.0%} of clips have transcripts"

    def test_speakers_identified(self, esther_ep1):
        speakers = esther_ep1.speakers
        assert len(speakers) >= 2, f"Expected >=2 speakers, got {len(speakers)}"

    def test_clips_are_valid_wav(self, esther_ep1):
        import wave
        clips = esther_ep1.clips_for()[:10]
        for clip in clips:
            with wave.open(clip["path"], "rb") as w:
                assert w.getnchannels() == 1
                assert w.getsampwidth() == 2
                assert w.getframerate() == 16000
                assert w.getnframes() > 0

    def test_daic_has_two_speakers(self, daic_300):
        speakers = daic_300.speakers
        assert len(speakers) == 2, f"DAIC-WOZ should have 2 speakers, got {len(speakers)}"

    def test_segment_durations_reasonable(self, esther_ep1):
        clips = esther_ep1.clips_for()
        durations = [c["duration"] for c in clips]
        avg = sum(durations) / len(durations)
        assert avg > 3, f"Average clip duration too short: {avg:.1f}s"
        assert avg < 30, f"Average clip duration too long: {avg:.1f}s"
