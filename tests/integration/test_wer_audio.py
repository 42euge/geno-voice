"""iter-117 — Real audio fixtures + faster-whisper integration.

Loads short .wav files from `tests/fixtures/wer/`, transcribes
each via `faster-whisper` (CPU-only `tiny` model), and asserts
the WER lands inside the per-fixture `[expected_wer_min,
expected_wer_max]` band.

Skips cleanly when faster-whisper can't load (no installed
package, no model cache, no network) — the CI host shouldn't
fail just because the heavyweight deps aren't available.

Pairs with iter-106's text-only fixtures: those exercise the
WER plumbing without audio; this exercises the same plumbing
through a real STT engine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_wer import compute_wer  # noqa: E402

CORPUS_PATH = ROOT / "tests" / "fixtures" / "wer" / "corpus.json"
FIXTURE_DIR = CORPUS_PATH.parent


# ---- Skip conditions -----------------------------------------------------


# Try to import faster-whisper at module load. If it fails, every test
# in this module is skipped — but pytest collection still succeeds.
try:
    from faster_whisper import WhisperModel
    _FW_AVAILABLE = True
except Exception as e:
    WhisperModel = None  # type: ignore
    _FW_AVAILABLE = False
    _FW_IMPORT_ERROR = str(e)


# Loading the model can fail if no cache + no network. We check
# at fixture-creation time so a single skip explains the failure.
@pytest.fixture(scope="module")
def stt_model():
    if not _FW_AVAILABLE:
        pytest.skip(
            f"faster-whisper not importable: {_FW_IMPORT_ERROR}"
        )
    try:
        return WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception as e:
        pytest.skip(f"faster-whisper model failed to load: {e}")


@pytest.fixture(scope="module")
def audio_fixtures():
    with CORPUS_PATH.open() as f:
        data = json.load(f)
    return data.get("audio_fixtures", [])


# ---- Tests --------------------------------------------------------------


def test_corpus_declares_audio_fixtures(audio_fixtures):
    """Sanity: the JSON has at least one audio fixture defined.
    Doesn't exercise STT — runs even when faster-whisper is
    unavailable so a corpus-deletion regression surfaces."""
    assert len(audio_fixtures) >= 1
    for entry in audio_fixtures:
        assert entry["audio_path"]
        assert entry["reference"]
        assert "expected_wer_min" in entry
        assert "expected_wer_max" in entry


def test_audio_files_exist_on_disk(audio_fixtures):
    """Sanity: each declared audio_path resolves to a real .wav.
    Doesn't transcribe — guards against a corpus update missing
    its companion file."""
    for entry in audio_fixtures:
        path = FIXTURE_DIR / entry["audio_path"]
        assert path.exists(), f"missing audio fixture: {path}"
        assert path.stat().st_size > 0


def _transcribe(stt_model, audio_path: Path) -> str:
    """Run STT and return concatenated text.

    iter-125: forced to greedy decoding (``beam_size=1,
    temperature=0``) so per-fixture WER is deterministic across
    runs. Without this, faster-whisper's default beam-search +
    temperature-fallback gives different transcripts on
    repeated runs of the same audio — catastrophic-band
    fixtures sometimes land at WER 0.2 (recovers) and sometimes
    1.0 (fails). Greedy decoding makes WER bands testable.
    """
    segments, _info = stt_model.transcribe(
        str(audio_path), language="en",
        beam_size=1, temperature=0,
    )
    # Generator — consume immediately.
    return " ".join(s.text for s in segments).strip()


def test_each_audio_fixture_lands_in_wer_band(stt_model, audio_fixtures):
    """The integration check. For each audio fixture:
      - Load the .wav
      - Transcribe via faster-whisper
      - Compute WER against the reference
      - Assert WER inside the recorded band
    """
    failures = []
    for entry in audio_fixtures:
        path = FIXTURE_DIR / entry["audio_path"]
        hyp = _transcribe(stt_model, path)
        wer = compute_wer(entry["reference"], hyp)
        if not (
            entry["expected_wer_min"] <= wer <= entry["expected_wer_max"]
        ):
            failures.append(
                f"{entry['name']}: WER {wer:.3f} outside "
                f"[{entry['expected_wer_min']}, "
                f"{entry['expected_wer_max']}] "
                f"(ref={entry['reference']!r}, hyp={hyp!r})"
            )
    assert not failures, "\n".join(failures)


def test_clean_audio_round_trips_below_30_pct_wer(stt_model, audio_fixtures):
    """Tighter assertion specifically on the clean audio fixture.
    Production-grade STT on synthesized speech should easily be
    under 30% WER; if it isn't, something is fundamentally
    broken (wrong model, audio corrupted, etc.)."""
    clean = next(
        (e for e in audio_fixtures if e["name"] == "clean_audio"),
        None,
    )
    if clean is None:
        pytest.skip("no clean_audio fixture in corpus")
    path = FIXTURE_DIR / clean["audio_path"]
    hyp = _transcribe(stt_model, path)
    wer = compute_wer(clean["reference"], hyp)
    assert wer < 0.30, (
        f"clean-audio WER too high: {wer:.3f}, "
        f"ref={clean['reference']!r}, hyp={hyp!r}"
    )
