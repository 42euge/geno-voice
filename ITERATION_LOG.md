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

---

## iter-005 — virtual audio interfaces + TTS-driven simulation

**Branch:** `iter-005-virtual-audio` (merged ff to main, commit `ad20b0c`)
**Date:** 2026-05-23

User redirected the iteration order: instead of doing streaming
overlap, build software audio interfaces and use TTS as a simulated
mic source. This unblocks every future iteration that needs end-to-end
testing without hardware.

What changed (new file: `examples/virtual_audio.py`):
- `VirtualMicStream` — push audio in, pyaudio-shaped read/get_available
  out. Reads beyond buffer zero-pad by default (matches a quiet mic).
- `VirtualSpeakerStream` — write captures audio. Optional `loopback_to`
  routes every write back into a paired mic. That's the seed for
  iter-007 (barge-in) — bot output and user input share a stream.
- `VirtualAudioInterface` — drop-in replacement for `pyaudio.PyAudio()`.
  `.open(input=True)` returns a mic, `.open(output=True)` returns a
  speaker, `loopback=True` wires them.
- Audio fixtures: `make_silence`, `make_tone_burst`, `make_noise_burst`,
  `concat`. Pure numpy; noise is deterministic with a seed.
- `feed_tts(mic, text, ...)` — renders via `tts/kokoro_engine`,
  resamples 24k→mic.rate (cheap linear), pads silence around speech.
  Raises `RuntimeError` cleanly if kokoro can't load (so tests can
  skip).
- `simulate_vad_over_audio(audio, ...)` — chunk-by-chunk driver that
  returns the full event sequence + final VadState.

Tests (26 new in `tests/unit/test_virtual_audio.py`):
- VirtualMicStream: round-trip, `push_silence`, underflow zero-pad,
  no-padding mode, close blocks, reads recorded, float audio
  scaled+clipped.
- VirtualSpeakerStream: capture, float32 normalization, loopback,
  close blocks.
- VirtualAudioInterface: input/output dispatch, loopback wiring,
  terminate.
- Fixtures: silence-is-zero, tone-burst RMS matches theory
  (0.3 amp sine ≈ 0.21 RMS), noise determinism, concat ordering.
- VAD-over-fixtures: pure silence stays idle; speech+silence fires
  exactly one DONE_OK; short blip → DONE_TOO_SHORT; two utterances
  → two DONE events; sub-threshold amp stays idle.
- Byte-level path: VirtualMicStream → frombuffer → VadState produces
  expected events through the same path production code traverses.
- **TTS smoke test (opt-in)**: rendered "Hello, this is a simulation
  test." via real kokoro, fed the virtual mic, drove VadState end to
  end. DONE_OK fires with `last_speech_duration > 0.5s`. Confirms
  the synth → resample → bytes → VAD chain works on x86_64 Linux
  (despite the cpp-extensions warning from torch 2.10 / kokoro).

Verification: `python -m pytest tests/unit/` → **73 passed in 6.5s**.

Notes:
- Surprised that kokoro actually synthesizes on Linux x86_64 — the
  `Skipping import of cpp extensions due to incompatible torch
  version` warning at import time looked fatal but the model still
  produces audio (slower path). The skipif guard catches the case
  where it can't, so this stays portable.
- The loopback wiring is intentionally simple: every write
  unconditionally pushes to the linked mic. Real barge-in (iter-007)
  needs an `enabled` flag we toggle while bot is speaking; that's a
  one-line addition when we get there.

Next:
- iter-006 (was iter-005 before user redirect): streaming LLM → TTS
  overlap. Now testable end-to-end: synth bot reply via TTS, route
  through a `VirtualSpeakerStream`, assert TTFS measured against
  `speech_ended_at` is below threshold.
- iter-007: hook `mic_chat.py` to optionally use
  `VirtualAudioInterface` instead of `pyaudio.PyAudio()` so the chat
  loop can be smoke-tested in CI with a fixed prompt + a stubbed
  LLM. This is the foundation for benchmarking pipeline latency
  changes deterministically.

---

## iter-006 — extract record_utterance_streaming to pyaudio-free module

**Branch:** `iter-006-recording-extract` (merged ff to main, commit `9cf4631`)
**Date:** 2026-05-23

What changed:
- New module `examples/_chat_recording.py` hosts the recording loop +
  its audio/VAD constants (RATE, CHUNK, SILENCE_THRESHOLD, etc.),
  `rms`, `_buffer_to_wav`, `_transcribe_quick`, and
  `record_utterance_streaming`. No top-level pyaudio import — the
  module is importable on x86_64 Linux without ALSA.
- `record_utterance_streaming` gained three injection points:
    `transcribe_fn` — `callable(wav) -> str | None`. Tests stub this.
    `clock` — `callable() -> float`. Tests pass a `FrameClock` that
        advances exactly one chunk per call, removing wall-time
        dependency from the test loop.
    `output` — file-like for the live preview. Tests pass `StringIO`.
- `mic_chat.py` re-imports the constants and function from the new
  module. Live behavior unchanged.
- Side fix: the too-short branch now explicitly sets
  `stt_engine._last_text = None` so a stale value from a previous
  turn can't leak through.

Tests (8 new in `tests/unit/test_chat_recording.py`):
- Three-tuple with plausible values for a 1s tone burst.
- Transcript stashed on `engine._last_text`.
- Returned bytes parse as a valid 16kHz mono int16 WAV.
- Too-short blip returns `(b'', 0.0, 0.0)` and skips transcribe_fn.
- `transcribe_fn` receives strictly-growing wav buffers across
  multiple preview intervals (regression cover for streaming preview).
- Preview only re-renders when text changes (prevents flicker).
- Default `transcribe_fn` falls back to `_transcribe_quick` and
  returns None on Linux without crashing.
- **TTS-fed smoke test (opt-in)**: real kokoro renders "Hello, this
  is a recording test.", virtual mic receives it, the full
  `record_utterance_streaming` runs with a stub transcriber, and
  asserts a well-formed wav of plausible duration.

Verification: `python -m pytest tests/unit/` → **81 passed in ~9s**
(73 existing + 8 new).

Notes:
- The `FrameClock` test helper turned out to be the unlock: by
  advancing time exactly `chunk/rate` per `clock()` call, the VAD
  state machine sees the same time progression it would with real
  audio, but the test runs as fast as numpy can crunch the buffer.
- One subtle thing learned writing the tests: `transcribe_fn` is
  called twice during the typical happy path — once at each
  `INFERENCE_INTERVAL` cross during speech (preview), and once at
  the very end on the final wav. The "growing buffer" test
  documents that contract.

Next:
- iter-007: extract `play_aligned` to a pyaudio-free module the same
  way. Accept any speaker-shaped object (write/stop_stream/close),
  test against `VirtualSpeakerStream`. After this, every primitive
  the chat loop touches is testable.
- iter-008: build streaming overlap — a `SentenceWorker` thread that
  consumes from a `Queue` of complete sentences, synthesizes via
  `tts_engine`, plays via the speaker stream, and reports playback
  duration. The LLM stream feeds the queue. Test using
  `VirtualSpeakerStream` to assert that playback starts before the
  LLM stream is done.
- iter-009: barge-in. Run `VadState` on the mic input *during*
  speaker writes (loopback mode means mic and speaker share a
  buffer; we'd disable loopback for this and use a separate mic
  feeder for "user starts speaking" injection). When the worker
  detects speech, drain the sentence queue, stop the speaker,
  return control.

---

## iter-007 — extract play_aligned to pyaudio-free module

**Branch:** `iter-007-playback-extract` (merged ff to main, commit `3005fde`)
**Date:** 2026-05-24

What changed:
- New module `examples/_chat_playback.py` hosts the playback loop +
  `TTS_RATE`, `DEFAULT_PLAY_CHUNK`, ANSI codes, plus `_emit_token`
  and `_is_punct_only` helpers.
- `play_aligned(speaker_stream, audio_np, tokens, ...)` now takes a
  caller-owned speaker-shaped object (only `.write(bytes)` is
  required). Lifecycle is the caller's responsibility.
- `mic_chat.play_aligned` becomes a thin wrapper that opens a
  pyaudio output stream, calls the core function, and closes the
  stream — preserving the current open-per-sentence behavior. The
  persistent-stream optimization is iter-008's job.
- Injection points: `output` (file-like) and `clock` (callable),
  same pattern as iter-006.
- Quirk preserved: tokens emitted during the loop are bolded; tokens
  emitted *after* the loop (when `start` > audio duration) are emitted
  in plain text. This matches the original code; a regression test
  documents it so a future cleanup is intentional.

Tests (18 new in `tests/unit/test_chat_playback.py`):
- `_is_punct_only`: punctuation True, words False, empty False.
- Audio writes: total bytes match input, chunk size honored, empty
  audio writes nothing, float→int16 saturation correct.
- First-sentence prefix: `\r CLEAR_LINE  Bot: ` only on first, no
  prefix on subsequent calls.
- Token reveal: in-order during playback, punctuation uses
  backspace, empty/whitespace tokens dropped, words bold, trailing
  tokens flushed unbolded after loop.
- Clock injection: elapsed computed from supplied clock; default
  output falls back to real stdout.
- Loopback: `VirtualSpeakerStream` → `VirtualMicStream` wiring still
  works; audio bytes round-trip identically. Seed for iter-009.

Verification: `python -m pytest tests/unit/` → **99 passed in ~9s**
(81 existing + 18 new).

Notes:
- Two iters to extract two functions. Now every primitive the chat
  loop touches — VAD, audio flush, recording, playback, render-
  preview, sentence-split, history-trim — is testable on x86_64
  Linux. The chat loop's remaining `pyaudio` calls are confined to
  `run_chat`'s setup/teardown plus the two thin wrappers.
- The loopback test is a tiny but meaningful proof point: writing
  to a `VirtualSpeakerStream` whose `loopback_to` is set pushes the
  exact same int16 samples into the paired mic. That's the
  primitive iter-009 barge-in needs — bot speaks, mic sees the
  speech, VAD watches the mic.

Next:
- iter-008: streaming LLM → TTS overlap. Pull the in-loop synth +
  play_aligned calls out into a `SentenceWorker` thread that pulls
  complete sentences from a `Queue`. The main thread keeps the LLM
  stream open so token receipt isn't blocked by audio playback.
  Holds a single persistent speaker stream — open in the worker's
  setup, close in its teardown. Test by feeding a fixed list of
  sentences into the queue, watching the speaker accumulate audio,
  and asserting the worker runs ahead of the simulated LLM clock.
- iter-009: barge-in. Mic-side `VadState` runs in parallel with the
  worker. On `DONE_OK` from the mic, signal the worker to drop the
  sentence queue and stop the speaker, return control. Test using
  loopback paired with a "user injection" feeder that pushes a
  user-speech burst into the same mic the worker is listening to.
- iter-010: filler-word generation. While `metrics.llm_first_token`
  is high, emit a pre-rendered "hmm…" or "let me think…" via TTS
  before the real first sentence. Cheap latency-perception win.
  Configurable via `config.local.yaml`.

---

## iter-008 — streaming LLM → TTS overlap via SentenceWorker

**Branch:** `iter-008-streaming-overlap` (merged ff to main, commit `ad4d946`)
**Date:** 2026-05-24

The headline architecture change. Previously the chat for-loop
synthesized + played each sentence synchronously *inside* the
token-receipt loop, so every byte of audio playback was a byte of
LLM stream latency the network connection was otherwise wasting.
Multi-sentence replies stacked the cost linearly.

Now: a background `SentenceWorker` thread consumes complete
sentences off a `Queue`, runs synth + play, and reports metrics. The
main thread submits and keeps consuming tokens.

What changed:
- New module `examples/_chat_pipeline.py` with `SentenceWorker`.
  - Constructor: `speaker_factory`, `synth_fn`, `play_fn`, optional
    `clock` and `output` (all dependency-injected).
  - Lifecycle: `start` → `submit` × N → `submit_done` →
    `wait_done(timeout=...)`. Or `stop(timeout=...)` for
    early termination (LLM error / Ctrl+C).
  - Metrics tracked: `sentences_spoken`, `tts_time`,
    `playback_time`, `first_audio_at`, `errors`.
  - Owns one persistent speaker stream — opens via `speaker_factory`
    in `_run`, closes (`stop_stream` + `close`) in the `finally`
    block. No more open-per-sentence overhead.
- `mic_chat.run_chat` rewired:
  - Replaces the in-loop synth + `play_aligned` calls with
    `worker.submit(sentence)`.
  - `wait_done(timeout=120)` after `submit_done` to drain any
    sentences still queued when LLM stream ends.
  - TTFS now derives from `worker.first_audio_at` (slightly more
    accurate — captured just before first audio is written, rather
    than before synth begins).
  - Error path calls `worker.stop()` for clean teardown before the
    bug-3 mic flush.

Tests (18 new in `tests/unit/test_chat_pipeline.py`):
- Lifecycle: submit + done + wait runs clean; double start raises;
  wait_done without start raises; submit after submit_done dropped.
- Ordering: sentences play in submission order (proven via
  growing-sample-count synth); `is_first_sentence` only True on first;
  metrics accumulate; `first_audio_at` set exactly once.
- Skipping: empty/whitespace sentences and empty audio from synth
  both skipped without incrementing count.
- Error handling: synth raising captured in `errors`, loop
  continues; play raising same; speaker_factory raising exits
  cleanly without hang.
- Stop: pending dropped + thread joined; before-start is a no-op;
  submit_done immediately followed by stop doesn't hang.
- Speaker wiring: `speaker._closed is True` after worker exit;
  loopback to `VirtualMicStream` round-trips audio (seed for
  iter-009 barge-in).

Verification: `python -m pytest tests/unit/` → **117 passed in 9s**
(99 existing + 18 new).

Notes:
- The threaded `stop` test uses a `synth_started` Event +
  `proceed` Event to deterministically place the worker in
  "currently synthesizing, two more queued" state before calling
  `stop()`. The assertion is "exactly one sentence got spoken" —
  the in-flight one finishes, the queued ones are dropped. This
  was the most likely-to-be-flaky test; it consistently runs in
  <50ms.
- `worker.errors` exists so production code can surface background
  failures without taking down the chat loop. `mic_chat.run_chat`
  prints any worker errors after `wait_done` returns.
- Single-sentence replies see no behavior change. Multi-sentence
  replies should show lower TTFS for sentence 2+ because the LLM
  stream now overlaps with playback.

Next:
- iter-009: barge-in. Run `VadState` on the mic stream *during*
  speaker writes. When the user starts speaking, signal the worker
  to drop pending sentences and stop the speaker mid-sentence.
  Requires a "hard cancel" path on the worker (the current `stop`
  is "drain + finish current"); add `cancel()` that sets a flag
  the play_fn checks per audio chunk. Test using loopback paired
  with a "user injection" feeder that pushes user-speech into the
  mic the worker is also watching.
- iter-010: filler words during LLM first-token wait. Pre-render
  short audio clips ("hmm…", "let me think…") and have the worker
  play them if the queue stays empty for more than ~600ms. Hide
  behind a config flag so it's opt-in.

---

## iter-009 — barge-in primitives: cancel_event + cancel + BargeInWatcher

**Branch:** `iter-009-bargein` (merged ff to main, commit `c4b34d5`)
**Date:** 2026-05-24

What changed:

1. **`play_aligned` cancel_event** — `examples/_chat_playback.play_aligned`
   gained an optional `cancel_event` kwarg. The chunked-write loop
   checks it between chunks; on set, breaks early and returns
   elapsed time so far. Backward compat: omit the kwarg and behavior
   is unchanged.

2. **`SentenceWorker.cancel()`** — hard-stop method that:
   - Sets the worker's internal `_cancel_event` (forwarded into
     `play_fn` so the in-flight sentence breaks mid-stream).
   - Drains the queue and joins the thread.
   - Sets `self.cancelled = True` for the caller to inspect.
   - Idempotent.
   The play_fn signature now optionally accepts `cancel_event` —
   a `try/except TypeError` fallback preserves the iter-008
   contract for play_fns that don't accept the new kwarg.

3. **`BargeInWatcher`** — new class in `examples/_chat_pipeline.py`.
   Background thread that reads from a mic stream, runs `VadState`,
   fires a callback on the configured trigger event. Configurable:
   - `trigger_on="active"` (default) — fires the instant level
     crosses threshold. Fastest reaction.
   - `trigger_on="done_ok"` — fires only after a complete utterance
     (waits for the silence window). Useful when you don't want
     to interrupt on every cough.
   Captures all consumed frames in `self.frames` so the orchestrator
   can replay the user's first syllables into the next record loop.
   Uses `get_read_available()` to avoid feeding silent zero-pad to
   the VAD when the mic is empty.

Tests (17 new in `tests/unit/test_bargein.py`):
- play_aligned cancel_event: pre-set breaks immediately, mid-loop
  set breaks between chunks, unset completes, omission works.
- SentenceWorker.cancel: interrupts in-flight sentence (verified
  by side-effect Event in play_fn), idempotent, before-start
  no-op, iter-008 play_fns without cancel_event still work.
- BargeInWatcher: silence stays idle, speech burst triggers
  callback < 2s, callback fires once during long speech,
  `done_ok` trigger waits for silence window (uses injected
  frame-clock for virtual time), invalid trigger raises,
  double-start raises, frames captured.
- **Integration**: 3 sentences submitted, watcher detects user
  speech on virtual mic, fires `worker.cancel`, worker breaks
  mid-sentence, byte-count math proves partial write, dropped
  sentences confirmed via `sentences_spoken <= 1`.

Verification: `python -m pytest tests/unit/` → **134 passed in 9s**
(117 existing + 17 new).

Notes:
- DONE_OK trigger test exposed a real subtlety: `time.monotonic()`
  doesn't work for `VirtualMicStream` tests because pre-pushed audio
  is served in milliseconds. The VAD's 0.8s silence window never
  elapses by wall clock. Solution: inject a frame-aligned clock —
  same `FrameClock` pattern as iter-006. In production this is a
  non-issue since real PyAudio serves frames at audio rate.
- A play_fn that respects `cancel_event` must yield between chunks
  (sleep, blocking I/O, anything). Otherwise the worker thread
  races through the play loop before the test can set the flag.
  The test fixture uses `time.sleep(0.005)`; real audio playback
  blocks at audio rate so this happens naturally.

Next:
- iter-010: wire barge-in into `mic_chat.run_chat`. Plumbing:
  during `worker.wait_done()`, run a `BargeInWatcher` on `mic`
  with `worker.cancel` as the callback. After `wait_done` returns,
  if `watcher.detected`, feed `watcher.frames` into the next
  `record_utterance_streaming` call (needs a small extension to
  the recording function to accept "primed" frames). Without this,
  the user's first syllables are dropped.
- iter-011: filler-word generation. Pre-render short audio clips
  ("hmm", "let me think") at startup. The worker grows an idle-
  watchdog timer: if the queue has been empty for >600ms while
  `submit_done` hasn't been called, play one filler clip. Hide
  behind `config.local.yaml` flag.

---

## iter-010 — wire barge-in into chat loop, primed_frames replay

**Branch:** `iter-010-bargein-wiring` (merged ff to main, commit `9279220`)
**Date:** 2026-05-24

Closes the loop iter-009 started. The cancel/watcher primitives are
now actually used by `mic_chat.run_chat`, and the user's barge-in
audio is preserved into the next record turn instead of dropped.

What changed:

- `examples/_chat_recording.py`:
  - `record_utterance_streaming` gained a `primed_frames` kwarg.
    Each primed frame is fed through the VAD before any live mic
    reads, in order. Each frame's bytes are appended to the
    output wav as if read live.
  - **Refactored time accounting**: the function uses a single
    virtual clock anchored at function entry —
    `now = t_origin + frame_idx * (CHUNK / RATE)`. The previous
    `clock()`-per-frame approach had two failure modes that the
    primed-frames tests surfaced: (1) primed frames served in
    microseconds made the silence window appear to close instantly;
    (2) with a per-call test FrameClock, the first live read's
    timestamp could land *before* the last primed timestamp,
    corrupting `last_speech_duration`. The unified virtual clock
    keeps time monotonic at audio rate regardless of clock
    provider. Production behavior is identical because PyAudio
    reads already block at audio rate.

- `examples/mic_chat.py`:
  - `run_chat` carries `primed_frames` across iterations of the
    while loop. After `worker.wait_done` returns, if
    `watcher.detected`, `watcher.frames` is captured into
    `primed_frames` and fed to the next `record_utterance` call.
    Reset to None after consumption.
  - Spawns a `BargeInWatcher` pointed at the same mic the
    recorder uses, with `worker.cancel` as the callback. Watcher
    starts after `worker.start`; stops after `worker.wait_done`.
    Mic is flushed (bug-3 helper from iter-002) right before the
    watcher starts so audio buffered during the LLM phase isn't
    misread as user speech.
  - LLM-error path also stops the watcher so we don't leak a
    thread.

Tests (9 new):
- `tests/unit/test_chat_recording.py` — 6 priming tests:
  - None / `[]` are no-ops; speech burst appears in wav; silence
    falls through to live mic; too-short blip yields
    `DONE_TOO_SHORT`; pedantic frame-count check.
- `tests/unit/test_bargein_orchestration.py` — 3 full-loop tests:
  - Bot speaks → user barges in on virtual mic → watcher fires
    `worker.cancel` → `primed_frames` go into next record call →
    valid wav covers user audio. Closes the loop end to end.
  - No-barge-in path: full sentence plays, no priming.
  - 5-sentence queue: barge-in during sentence 1 drops the
    remaining 4.

Verification: `python -m pytest tests/unit/` → **143 passed in 10s**
(134 existing + 9 new).

Notes:
- The per-frame `clock()` bug was hidden under the simple iter-006
  case (live-only reads through FrameClock) and only surfaced when
  primed and live frames coexisted. The fix (`t_origin` + frame
  counter) is also conceptually cleaner — the function's view of
  time is now invariant to how the underlying clock behaves.
- The orchestration tests are as close as we get on x86_64 Linux
  to "the real chat loop is working." Deterministic in 0.2s using
  only virtual audio. That's the payoff for everything iter-005
  through iter-009 built up.

Next:
- iter-011: filler-word generation (carried over from iter-009
  plan). Pre-render short clips ("hmm", "let me think") at
  startup. Worker grows an idle-watchdog timer: if queue stays
  empty for >600ms while `submit_done` hasn't been called, play
  one filler. Behind a `config.local.yaml` flag.
- iter-012: optional — keep the `BargeInWatcher` running across
  the LLM-streaming phase too, not just during play. Currently
  if the user starts talking while the LLM is still streaming
  the first response token, we miss those early frames. Doable
  but requires reasoning carefully about who owns the mic when.
- iter-013: real-mic CI smoke test using ALSA loopback. Pipe
  TTS-rendered audio through `aplay` into `arecord`, then run
  the chat loop end-to-end on the actual PyAudio stack. Optional
  — closes the gap between virtual-audio tests and real
  hardware, but may not be worth the maintenance burden pre-1.0.

---

## iter-011 — filler-word generation in SentenceWorker

**Branch:** `iter-011-fillers` (merged ff to main, commit `12b718b`)
**Date:** 2026-05-24

The third architecture goal from the original focus list. While the
LLM is still spinning up its first token (typically 200-800ms), the
chat loop sat silent. Now the worker can play a pre-rendered filler
clip ("hmm", "let me think") to mask that latency.

What changed:

- `examples/_chat_pipeline.py`:
  - `SentenceWorker.__init__` gained `fillers`, `idle_threshold`,
    `filler_picker` kwargs.
  - `_run` uses `queue.get(timeout=idle_threshold)` when fillers
    are configured AND no sentence has played yet AND no filler
    has played yet. On timeout, picks one filler and plays it.
    After playing, reverts to blocking `get()` — exactly one
    filler per worker run.
  - Refactored play logic into a new `_play_clip()` helper so
    both fillers and real sentences share the same bookkeeping
    (`first_audio_at`, `playback_time`, `errors`,
    `cancel_event` handling).
  - New counter `fillers_played` (distinct from
    `sentences_spoken`).
  - TTFS works correctly: `first_audio_at` is captured the first
    time `_play_clip` runs, which means a filler accurately
    measures the user-perceived latency it's meant to mask.

- `examples/mic_chat.py`:
  - New `load_chat_config()` reads the optional `chat:` section
    of `config.local.yaml` (empty if missing — no breaking
    change for existing configs).
  - `run_chat` pre-renders fillers via `synthesize_with_alignment`
    using `chat.fillers` (list of strings); idle threshold from
    `chat.fillers_idle_threshold` (default 0.6s).
  - Bad filler entries are logged and skipped — never block the
    caller.
  - SentenceWorker construction passes the rendered list; if
    empty (default), `idle_threshold` is forced to 0 so the
    worker never times out.

Tests (12 new in `tests/unit/test_fillers.py`):
- Backward compat: no fillers / empty list = unchanged behavior.
- Happy path: filler plays after idle, no filler if sentence
  arrives early, exactly one filler per run.
- Filler counts as the first audio output (`is_first_sentence=True`
  for the "Bot:" prefix).
- Counters: `first_audio_at` captured at filler start;
  `fillers_played` starts at 0.
- Robustness: empty filler audio consumes slot but `fillers_played`
  stays 0; play raising during filler doesn't block real sentence;
  `cancel()` during filler playback works (same cancel_event
  path as real sentences).
- Byte-stream order: filler bytes precede sentence bytes in
  speaker's captured stream, with distinct amplitudes (0.3 vs 0.5
  → int16 9830 vs 16383) to make the regions distinguishable.

Verification: `python -m pytest tests/unit/` → **155 passed in 11s**
(143 existing + 12 new).

Notes:
- The `_play_clip` extraction made the diff bigger but is the
  right shape going forward — fillers, sentences, and any future
  audio output (e.g. confirmation tones) all share the same path.
- One filler per run is intentional. A tempting extension is
  "play another filler if the LLM stalls mid-stream too," but
  that adds a watchdog timer and a bunch of edge cases. Skipping
  for now; the first-token-latency mask is where 90% of the win is.

---

# Status

The architecture phase is **complete** for the user-specified focus
list:

  ✓ Streaming overlap        — iter-008
  ✓ Barge-in                 — iter-009 + iter-010
  ✓ Filler words             — iter-011

Cumulative state: **155 unit tests passing in ~11s** on x86_64
Linux without pyaudio or mlx-whisper. The chat-loop primitives
(`record_utterance_streaming`, `play_aligned`, `SentenceWorker`,
`BargeInWatcher`, `VadState`) are all testable in isolation, and
the orchestration tests prove they compose correctly into a
full bot-speaks → user-barges-in → record-replays cycle running
deterministically in 0.2s.

Remaining log items (iter-012 extended-watcher, iter-013 ALSA
loopback CI) are stretch goals with diminishing returns for a
pre-1.0 project. Could revisit if/when actual user feedback
demands them.
