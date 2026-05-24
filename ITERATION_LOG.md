# Iteration Log

Each entry summarizes one /loop iteration: what was learned, what changed, what is next.

## Known Bugs (from initial diagnosis)

1. **Duplicate "Bot:" lines** — `mic_chat.py:377` prints `[N] waiting...` with `end=""` but never gets cleared before bot output. Multiple "Bot:" lines appear during multi-sentence responses.
2. **Timer shows absurd values (3201756ms)** — `mic_chat.py:470` computes `llm_total` after all playback completes (includes TTS + playback time, not just LLM).
3. **No error recovery after LLM failure** — `mic_chat.py:479-482` does `messages.pop(); continue` without flushing the mic buffer. STT accumulates garbage after DNS errors.
4. **Excessive live transcription reprints** — `mic_chat.py:216` uses `\r` which breaks when preview text wraps terminal width.

## Goals

- Fix all 4 bugs above
- Add unit tests for the pipeline logic (sentence splitting, metrics calculation, error recovery)
- Refactor mic_chat.py into testable components (separate display, timing, error handling)
- Implement streaming LLM → TTS overlap (start TTS while LLM still streaming)
- Add filler-word generation to mask latency (configurable "thinking" sounds)
- Improve turn-taking (allow barge-in / interruption during bot speech)

---

## iter-001 — fix bugs #1 and #2, extract testable helpers

**Branch:** `iter-001-bugs-1-2` (merged ff to main, commit `1a2656a`)
**Date:** 2026-05-23

What changed:
- Bug #1 (duplicate "Bot:" lines) fixed: cleared the `[N] waiting...` row
  before printing the `Bot:` prefix in `play_aligned`. Single-line edit
  in `examples/mic_chat.py`.
- Bug #2 (absurd `llm_total` values) fixed: stamped
  `llm_stream_done_at` immediately after the streaming for-loop and used
  that for `metrics.llm_total`. Previously the timer was sampled after
  the trailing TTS + playback for the partial last sentence, which is
  why values like 3201756ms appeared.
- Refactored the small testable parts of `mic_chat.py` into
  `examples/_chat_helpers.py` (stdlib only, no audio imports):
  `split_complete_sentences`, `trim_history`, `TurnTimings`,
  `format_preview_line`. `mic_chat.py` now imports and uses these.
- Added `tests/unit/test_chat_helpers.py` with 22 tests covering
  sentence splitting (empty, no-terminator, multi, mid-sentence period),
  history trim (empty, short, long, no-system, mutation), `TurnTimings`
  contracts (first-token-once, llm_total stays put after stream done,
  TTFS recorded once), and preview-line truncation.

Verification: `python -m pytest tests/unit/` → **22 passed in 0.02s**.

Notes:
- Pre-existing issues NOT touched: `tests/test_session.py` collection
  error, `tests/e2e/` fixture/server errors. These also fail on main
  prior to this iteration, so they're out of scope.
- `format_preview_line` is groundwork for bug #4 (next iteration) —
  it's tested here but not yet wired into the live preview render.

Next:
- iter-002: bug #3 (mic buffer flush after LLM error) — requires
  refactoring the audio stream to be drainable from a test, or at
  minimum extracting `flush_pending_audio(stream)` and unit-testing
  that it consumes available chunks without blocking.
- iter-003: bug #4 (preview wrap) — wire `format_preview_line` into
  `record_utterance_streaming` and verify with a fake stdout test.
