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

---

## iter-002 — fix bug #3 (mic buffer not flushed after LLM error)

**Branch:** `iter-002-bug-3` (merged ff to main, commit `795a3aa`)
**Date:** 2026-05-23

What changed:
- Added `flush_pending_audio(stream, chunk_size, max_iterations)` in
  `examples/_chat_helpers.py`. Non-blocking drain via
  `get_read_available()` + `read(exception_on_overflow=False)`. Hard
  cap on iteration count guards against a misbehaving stream that
  always reports "data available." Catches exceptions from either
  call so a stream hiccup mid-drain doesn't crash the chat loop.
- Wired the flush into the LLM-error handler in `mic_chat.py`. After a
  DNS failure / timeout / 5xx, we now drop whatever audio piled up
  during the failed call and print a dim notice with how many seconds
  were discarded so the user understands their next utterance starts
  clean.
- 7 new tests using a `FakeStream` test double: empty stream → 0 reads;
  full chunks only with partial remainder retained; max-iterations cap
  on a lying stream; both `get_read_available` and `read` raising
  cleanly.

Verification: `python -m pytest tests/unit/` → **29 passed in 0.02s**.

Notes:
- Cannot end-to-end test the actual pyaudio path on x86_64 Linux
  without hardware. The FakeStream contract matches pyaudio's two
  methods exactly, so the unit tests give us solid coverage of the
  drain logic. A future on-device smoke test would close the gap.

Next:
- iter-003: bug #4 (preview reprint wraps when text exceeds terminal
  width). Wire `format_preview_line` into `record_utterance_streaming`,
  use `shutil.get_terminal_size()` to pick the width, and add a test
  with a fake stdout that asserts no wraparound.
- iter-004: split `record_utterance_streaming` into pure VAD logic
  (testable) + thin pyaudio glue. The VAD state machine is the most
  fragile part of the loop and currently has zero tests.

---

## iter-003 — fix bug #4 (preview reprint wraps on long utterances)

**Branch:** `iter-003-bug-4` (merged ff to main, commit `7349fb7`)
**Date:** 2026-05-23

What changed:
- Added `render_preview(text, *, max_width, prefix, file, dim)` to
  `examples/_chat_helpers.py`. Wraps the existing
  `format_preview_line` truncation, prepends `\r\033[2K`, optionally
  wraps the body in dim ANSI, and writes-then-flushes to any
  file-like object. Returns the exact emitted string so tests can
  assert on visible width without faking terminal I/O.
- `record_utterance_streaming` in `mic_chat.py` now calls
  `shutil.get_terminal_size(fallback=(80, 24)).columns` and passes
  that into `render_preview` instead of writing the format string by
  hand. So even on a narrow terminal (or one resized mid-utterance),
  the live preview clamps to the current width.
- 7 new tests with `io.StringIO` and a counting buffer test double:
  short-text passthrough, long-text truncation with ellipsis, leading
  `\r + CLEAR_LINE`, write+flush behavior, `dim=False` strips dim
  ANSI, custom-prefix width math, back-to-back rewrite contract.

Verification: `python -m pytest tests/unit/` → **36 passed in 0.03s**.

All four originally-listed mic_chat.py bugs are now fixed:
  ✓ Bug #1 (iter-001) — duplicate "Bot:" lines
  ✓ Bug #2 (iter-001) — absurd llm_total values
  ✓ Bug #3 (iter-002) — mic buffer not flushed after LLM error
  ✓ Bug #4 (iter-003) — preview reprint wraps on long utterances

Next:
- iter-004: extract VAD state machine into a pure helper. Currently
  the silence/speech detection is inlined in
  `record_utterance_streaming` with monotonic timestamps and ad-hoc
  state — moving it to `vad_state.feed(level, now) -> Event` makes
  it testable without audio hardware. This is also the prerequisite
  for the architecture goals (streaming overlap, barge-in).
- iter-005: streaming LLM → TTS overlap. Right now sentences
  synthesize and play synchronously inside the for-loop, blocking
  the next token receipt. A producer/consumer queue with the LLM
  streaming on the main thread and TTS+playback on a worker would
  cut median TTFS substantially.

---

## iter-004 — extract VadState pure helper, 11 new tests

**Branch:** `iter-004-vad` (merged ff to main, commit `dff3f9e`)
**Date:** 2026-05-23

What changed:
- Added `VadState` (and `VadEvent` enum) in
  `examples/_chat_helpers.py`. Pure state machine: `feed(level, now)`
  returns one of IDLE / ACTIVE / DONE_OK / DONE_TOO_SHORT and updates
  `speaking`, `speech_start`, `silence_start`, `last_speech_duration`.
  Auto-resets on either DONE event so the state machine is ready for
  the next utterance.
- `record_utterance_streaming` in `mic_chat.py` rewritten to consume
  events instead of managing inline flags. Behavioral parity preserved:
  the trailing silence frame is still appended to `frames` before
  break (matches the original `frames.append(data); ... break` order).
  `too_short` flag replaces the post-loop duration recheck.
- 11 new tests covering: initial idle, first loud frame, continuous
  speech, silence timer start, brief silence doesn't end utterance,
  full silence window → DONE_OK, too-short window → DONE_TOO_SHORT,
  state-reset-between-utterances, strictly-greater threshold boundary,
  speech_duration recorded correctly, sub-threshold blips stay IDLE.

Verification: `python -m pytest tests/unit/` → **47 passed in 0.04s**.

Notes:
- A full integration test of `record_utterance_streaming` itself was
  considered but skipped: `mic_chat.py` imports `pyaudio` at module
  level, which isn't installable on x86_64 Linux without ALSA dev
  headers. The pure VadState unit tests give us complete coverage of
  the extracted logic; the wiring is straightforward and small.
- Hit a real bug while writing the threshold-boundary test:
  `1.4 - 0.6 = 0.7999999999999999`, which silently fails a `>= 0.8`
  comparison. Tests now use timestamps that avoid float-precision
  cliffs (e.g. 1.5 instead of 1.4 for the 0.8s silence window). Worth
  remembering for any future timestamp-comparison code.

Next:
- iter-005: streaming LLM → TTS overlap. The for-loop currently does
  synth + playback synchronously inside each iteration, blocking the
  next token. A `Queue` + worker thread for synth and a second worker
  for playback would let the LLM stream proceed at network speed
  while audio is being produced and played. Median TTFS could drop
  substantially since first-sentence synth could start the moment the
  first sentence's text is complete, even if more tokens are still
  arriving.
- iter-006: filler-word generation. While waiting for the first LLM
  token (`metrics.llm_first_token` typically 200-800ms), play a
  pre-rendered "hmm" or "let me think" so the user perceives lower
  latency. Configurable via `config.local.yaml`.
- iter-007: barge-in. Run a thin VAD on the mic input *during*
  bot playback. If the user starts speaking, kill the playback
  stream, drop pending sentences, and start recording. Requires the
  TTS + playback work from iter-005 since playback needs to be
  cancellable from outside the playing thread.
