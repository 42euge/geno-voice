# Voice-capture tuning (living doc)

A data-driven log of the geno-voice **capture / VAD / latency** system,
grounded in real user recordings rather than assertion. Seeded by iter-189.

The live capture path lives in `client/voice-capture.js`
(`ContinuousListener`): an RMS gate + a 200ms speech-onset debounce + an
800ms silence timeout. The desktop app vendors a copy, so changes here are
the **source** — the operator re-syncs and rebuilds to ship. Keep changes
parameter-driven so they port cleanly.

## Ground-truth corpus

Real sessions captured from the desktop app land in
`fixtures/recordings/*.wav` with sibling `.json` metadata
(`click_to_capture_ms`, `peak_rms`, `frames`, `sample_rate`). They are
**not committed** (large binary captures) — they are rsync'd onto the loop
host. The headless harness `fixtures/replay_vad.py` replays each recording
through a faithful Python port of the `ContinuousListener` state machine
for a given parameter set — no mic, no GUI, no browser.

```
python fixtures/replay_vad.py --threshold 0.006     # human-readable table
python fixtures/replay_vad.py --json                # machine-readable
```

`tests/integration/test_vad_recordings.py` turns every recording into a
regression test (the data flywheel): the more the user talks to the app,
the more fixtures land, and each must keep detecting its speech at the
production threshold. It **skips cleanly** when the corpus is absent, and
the harness logic is covered fast and deterministically by
`tests/unit/test_replay_vad.py` over synthetic WAVs.

## Per-recording RMS stats (seed corpus, threshold 0.006, frame 1024)

| Recording | dur | sample_rate | meta peak_rms | replay peak | mean RMS | median RMS | % over | onsets | speak frames | latency (ms) |
|-----------|-----|-------------|---------------|-------------|----------|------------|--------|--------|--------------|--------------|
| voice-20260617-122716 | 17.3s | 44100 | 0.03  | 0.0408 | 0.00286 | 0.00036 | 17.1% | 2 | 232 | 3913 |
| voice-20260617-123829 | 64.6s | 44100 | 0.073 | 0.0806 | 0.00161 | 0.00034 | 5.4%  | 2 | 220 | 3466 |
| voice-20260617-131451 | 9.3s  | 44100 | 0.037 | 0.0530 | 0.00267 | 0.00030 | 11.0% | 1 | 158 | 5127 |
| voice-20260617-135015 | 1115.8s | 44100 | 0.036 | 0.0594 | 0.00047 | 0.00024 | 0.7%  | 5 | 546 | 3090 |

Key observations:

- **Huge signal/silence separation.** Median RMS (≈ the silence floor)
  sits at ~0.0003 across every recording, while speech peaks reach
  0.04–0.08. The 0.006 threshold sits an order of magnitude above the
  floor and well below speech peaks — safe with wide margin.
- **Threshold 0.006 recovers all four; 0.015 drops one.** At the upstream
  default of 0.015, the long far-field session `voice-20260617-135015`
  (mean RMS 0.00047) detects **zero** onsets — only 0.1% of its frames
  clear the gate. At 0.006 it detects 5 onsets / 546 speaking frames.
  This is the empirical justification for the desktop client's lowered
  threshold, now pinned by `TestThresholdRegression`.
- **Latency is the dominant remaining bug.** `click_to_capture_ms` is
  **3.1–5.1s** across the corpus. The replay sees healthy speech because
  it replays the *whole* recording; the live client only starts capturing
  after getUserMedia + AudioWorklet cold-start, so it sees only whatever
  the user says *after* that 3–5s dead window. The replay quantifies the
  ceiling; the live path forfeits the opening of every utterance.

## Findings & backlog (prioritized)

1. **[latency] Pre-warm the capture pipeline.** The 3–5s `click_to_capture_ms`
   is the top user-facing bug. `getUserMedia` + `audioContext.resume()` +
   `audioWorklet.addModule()` are all cold-started inside
   `ContinuousListener.start()`. Prototype pre-warming: create the
   `AudioContext` and `addModule()` the worklet at app load (or on first
   user gesture), keep a muted stream warm, so `start()` only flips the
   `active` flag. Measure the new `click_to_capture_ms` against the corpus
   baseline. **Cannot be replayed** (it is a cold-start cost, not a signal
   property) — needs an on-device timing harness, but it is the highest-value
   item.
2. **[latency] Rolling pre-roll buffer.** Even with a warm context, keep a
   short ring buffer (e.g. 1–2s) of pre-onset audio so the committed
   segment includes the speech that arrived during the debounce/onset
   window. Today `_speechCandidate` discards sub-debounce audio; a pre-roll
   would recover clipped utterance openings. Testable via replay: assert the
   first segment's `onset_ms` moves earlier.
3. **[threshold] Confirm 0.006 holds as the corpus grows.** The regression
   test guards against silent regressions. Re-run `replay_vad.py` each lap
   on any newly-synced recordings; if a new recording misses at 0.006,
   investigate (gain stage? far-field? noise floor?) before lowering further.
4. **[gain] Software gain experiment.** `replay_vad.py --gain N` models a
   pre-amplification stage. Sweep gain vs. false-trigger rate on the silence
   floor: a modest gain (1.5–2×) might let the threshold stay conservative
   while recovering the quietest far-field frames. Pick the setting that
   maximizes onset recovery across the corpus without lifting silence-floor
   frames over the gate.
5. **[debounce] Tune onset debounce against clipping.** The 200ms debounce
   trades onset latency for false-trigger immunity. With the wide signal
   separation observed, a shorter debounce (e.g. 100ms) may be safe and
   would clip less of the utterance opening. Sweep via
   `--debounce-ms` and measure onset timing + false triggers on silence.
6. **[silence] Right-size the silence timeout.** 800ms may over-split or
   over-merge turns. The seed corpus shows clean multi-segment splits;
   revisit if new recordings show truncated or run-on turns.

## Methodology notes

- The replay state machine mirrors the JS faithfully: below-threshold
  frames clear the onset candidate; the candidate must hold over-threshold
  for **> debounce_ms** of consecutive frames before committing; while
  speaking, each below-threshold frame advances a silence clock and the
  segment ends once silence reaches `silence_ms`; a committed segment
  shorter than `min_speech_ms` is dropped.
- Frames are non-overlapping 1024-sample windows (the live worklet/
  scriptProcessor buffer size), RMS = `sqrt(mean(s^2))`. A trailing partial
  frame is included, matching the client processing whatever the last
  buffer holds.
- "Known speech would trigger" = the recording's metadata `peak_rms`
  clears the gate **and** the state machine committed ≥1 onset on replay.
  Recordings without metadata fall back to the onset count alone.
