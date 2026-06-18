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

To compare a whole grid of values in one run, sweep a single parameter
across the corpus (added iter-190). It replays the whole corpus once per
value and aggregates detection — the comparison table the tuning backlog
below keeps asking for, instead of N hand-run single-param invocations:

```
python fixtures/replay_vad.py --sweep threshold --sweep-values 0.004,0.006,0.015
python fixtures/replay_vad.py --sweep gain --sweep-values 1.0,1.5,2.0,3.0
python fixtures/replay_vad.py --sweep debounce_ms --sweep-values 100,200,300 --json
```

When two parameters *interact* — e.g. more gain lifts quiet speech over the
gate, so the best threshold shifts — a single-axis sweep is too coarse to
pick the joint operating point. A **2-D grid** (`--grid`, iter-192 — backlog
item 4) replays the corpus once per cell of two axes and reports every cell,
row-major (first axis = rows, second = columns):

```
python fixtures/replay_vad.py --grid threshold,gain \
    --grid-values-a 0.004,0.006,0.010,0.015 --grid-values-b 1.0,1.5,2.0
python fixtures/replay_vad.py --grid debounce_ms,preroll_ms \
    --grid-values-a 100,200 --grid-values-b 0,256,512 --json
```

A **pre-roll buffer** (`--preroll-ms`, iter-191 — backlog item 2) recovers
the soft attack of an utterance that the live client discards: every
sub-threshold frame before a speech onset is thrown away today, so the
committed segment starts at the debounce-committed frame and clips the
quiet ramp-up. Pre-roll keeps the last N ms of pre-onset audio and prepends
it, pulling the emitted `onset_ms` earlier — clamped to the recording start
and the previous segment's end (segments never overlap). `0.0` (the
default) reproduces today's clip-the-opening behaviour, so it is a no-op
until wired into the client. It moves onset *timing*, not onset *count*, so
inspect `onset_ms` (`--json`) rather than the aggregate sweep totals:

```
python fixtures/replay_vad.py --preroll-ms 256                 # single run
python fixtures/replay_vad.py --preroll-ms 256 --json          # see onset_ms shift
```

Each row reports, across the corpus: `trig` (how many recordings'
known speech would trigger), `min_onsets` (the worst single recording —
the floor a sweep wants to *maximize*, since one missed recording is a
real miss even when the total looks healthy), `onsets`/`speak_frames`
totals, `mean_over` (mean %-of-frames-over-threshold), and `onset1`
(iter-197 — the onset-*timing* aggregate: the mean of each recording's
**first** emitted `onset_ms`, averaged only over recordings that detected
speech). A smaller `onset1` means speech is captured earlier in the
recording; missed recordings are excluded so a miss can't masquerade as
great timing (it would otherwise fold in a 0ms onset). This is the
aggregate the onset-shaping knobs (`debounce_ms`, `preroll_ms`) move —
the count aggregates stay flat while timing shifts — so a single sweep now
shows timing moving earlier without hand-inspecting each recording's
`--json`. iter-198 adds the companion `onset1_max` column: the *latest*
first onset across detected recordings — the worst-case ceiling a sweep
wants to *minimize*. It is to `onset1` what `min_onsets` is to `onsets`:
the mean can look great while one recording is captured far too late, and
that single bad onset is a real regression the mean would hide.

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

## Parameter sweeps over the seed corpus (iter-190)

Run with `fixtures/replay_vad.py --sweep`. All sweeps hold the other
parameters at their defaults (frame 1024). `trig` = recordings whose
known speech would trigger; `min_onsets` = the worst single recording.

**Threshold** (`--sweep threshold`):

| threshold | trig | min_onsets | total onsets | speak frames | mean %over |
|-----------|------|------------|--------------|--------------|------------|
| 0.004 | 4/4 | 1 | 12 | 1353 | 11.4% |
| 0.006 | 4/4 | 1 | 10 | 1156 | 8.5%  |
| 0.010 | 4/4 | 1 | 8  | 623  | 5.6%  |
| 0.015 | 3/4 | 0 | 3  | 203  | 3.4%  |
| 0.020 | 1/4 | 0 | 1  | 62   | 1.8%  |

The cliff is between **0.010 and 0.015**: at 0.015 the far-field
`voice-20260617-135015` drops to `min_onsets=0` (a real miss) and `trig`
falls to 3/4. 0.006 keeps a full-corpus floor (`min_onsets=1`) with
margin — going down to 0.004 buys more frames but no new triggers, so it
risks lifting silence-floor frames over the gate for no detection gain.
**0.006 is confirmed as the right operating point** for this corpus.

**Gain** (`--sweep gain`, threshold 0.006):

| gain | trig | total onsets | speak frames | mean %over |
|------|------|--------------|--------------|------------|
| 1.0 | 4/4 | 10 | 1156 | 8.5%  |
| 1.5 | 4/4 | 12 | 1353 | 11.4% |
| 2.0 | 4/4 | 16 | 1557 | 13.0% |
| 3.0 | 4/4 | 18 | 2145 | 16.1% |

Gain monotonically lifts onsets and speaking frames. Since the silence
floor (~0.0003) is still ~3× below the 0.006 gate even at gain 2.0
(0.0006), modest gain recovers more speech without raising `%over` into
false-trigger territory. **A 1.5–2.0× gain is a safe, cheap recovery
lever** that backlog item 4 proposed — the sweep now quantifies it.
(Equivalent to lowering the threshold proportionally; gain is the knob
the worklet can apply directly.)

**Onset debounce** (`--sweep debounce_ms`, threshold 0.006):

| debounce_ms | trig | total onsets | speak frames |
|-------------|------|--------------|--------------|
| 100 | 4/4 | 11 | 1322 |
| 150 | 4/4 | 10 | 1201 |
| 200 | 4/4 | 10 | 1156 |
| 300 | 3/4 | 6  | 568  |

100–200ms all keep `trig=4/4`; the current **200ms is safe and 100ms
recovers slightly more speaking frames** (clips less utterance opening),
consistent with the wide signal separation. **300ms is too aggressive** —
it drops a recording. Backlog item 5's "100ms may be safe" hypothesis is
supported by the data; a follow-up should validate onset *timing* (not
just count) before changing the client.

## Pre-roll buffer over the seed corpus (iter-191)

`--preroll-ms` pulls each utterance's first `onset_ms` earlier by up to the
requested window (clamped to recording start / previous segment end). The
onset *count* is unchanged — pre-roll recovers the clipped opening of a
segment that was already detected. First-segment `onset_ms` at threshold
0.006, frame 1024:

| Recording | preroll 0 | preroll 256ms | preroll 512ms |
|-----------|-----------|---------------|---------------|
| voice-20260617-122716 | 232.2ms | 0.0ms (start clamp) | 0.0ms |
| voice-20260617-123829 | 2995.4ms | 2740.0ms | 2484.5ms |
| voice-20260617-131451 | 1370.0ms | 1114.6ms | 859.1ms |
| voice-20260617-135015 | 1532.5ms | 1277.1ms | 1021.7ms |

Every recording's opening moves earlier (the first clamps to 0 — its speech
begins ~232ms in, so a 256ms pre-roll reaches the recording start). The
committed segment's frame count rises correspondingly, so the prepended
audio is real recovered speech, not silence padding. **A 256–512ms pre-roll
recovers ~250–510ms of clipped utterance opening per turn** with no risk to
detection (it cannot create or drop onsets) and no overlap (the previous-
segment clamp is enforced and tested). This is the cheap, replay-validated
half of the latency story: pre-warming (item 1) recovers the cold-start
dead window; pre-roll recovers the debounce/onset clip on top.

## 2-D threshold × gain grid over the seed corpus (iter-192)

Run with `fixtures/replay_vad.py --grid threshold,gain`. The grid replays
the corpus once per cell and exposes the *interaction* a single-axis sweep
hides. `--grid-values-a 0.004,0.006,0.010,0.015 --grid-values-b 1.0,1.5,2.0`:

| threshold ↓ / gain → | 1.0 | 1.5 | 2.0 |
|----------------------|-----|-----|-----|
| **0.004** | trig 4/4, onsets 12 | 4/4, 16 | 4/4, 18 |
| **0.006** | trig 4/4, onsets 10 | 4/4, 12 | 4/4, 16 |
| **0.010** | trig 4/4, onsets 8  | 4/4, 7  | 4/4, 10 |
| **0.015** | **trig 3/4, min_onsets 0** | 4/4, 8 | 4/4, 7 |

**The key interaction:** at threshold 0.015 (the old under-detecting
default) the far-field `voice-20260617-135015` misses entirely
(`trig=3/4, min_onsets=0`) at unity gain — but **1.5× gain recovers it**
(`trig=4/4, min_onsets=1`). Gain and threshold trade off: a louder signal
lets a stricter gate still catch quiet far-field speech. A single-axis
threshold sweep at gain 1.0 would conclude 0.015 is unsafe; the grid shows
0.015 + 1.5× gain is as safe as 0.006 at unity. The chosen operating point
(threshold 0.006, gain 1.0) keeps `trig=4/4` with margin and avoids relying
on a gain stage that isn't wired into the client yet (backlog item 4), but
the grid documents that **a future gain stage would let the threshold rise
back toward 0.010–0.015 without losing the far-field recording** — useful if
a lower threshold ever proves to admit noise on a noisier corpus. Detection
is monotone along both axes (lower threshold ⇒ ≥ onsets; higher gain ⇒ ≥
onsets), pinned by `TestGridSweep` in
`tests/integration/test_vad_recordings.py`.

## Onset-timing aggregate over the seed corpus (iter-197)

Before iter-197 the sweep/grid only aggregated onset *counts*
(`total_onsets`, `min_onsets`), so validating that an onset-shaping knob
moves speech-capture *earlier* meant hand-reading each recording's
`onset_ms` out of `--json`. iter-197 adds `mean_first_onset_ms` (the
`onset1` column) to `SweepPoint`, so the timing lever is now visible in a
single sweep pass. The two onset-shaping knobs, swept over the seed corpus
at threshold 0.006, frame 1024:

```
python fixtures/replay_vad.py --sweep debounce_ms --sweep-values 100,200,300
```
| debounce_ms | trig | min_onsets | onsets | onset1 (mean first) |
|-------------|------|------------|--------|---------------------|
| **100** | 4/4 | 1 | 11 | **1271.3ms** |
| 200 (default) | 4/4 | 1 | 10 | 1532.5ms |
| 300 | **3/4** | **0** | 6 | 2190.4ms |

```
python fixtures/replay_vad.py --sweep preroll_ms --sweep-values 0,256,512
```
| preroll_ms | trig | onsets | onset1 (mean first) |
|------------|------|--------|---------------------|
| 0 (default) | 4/4 | 10 | 1532.5ms |
| 256 | 4/4 | 10 | 1282.9ms |
| 512 | 4/4 | 10 | 1091.3ms |

**The debounce-timing evidence backlog item 5 was waiting for:** lowering
the onset debounce from 200ms→100ms pulls the mean first onset **~261ms
earlier** (1532.5→1271.3ms) while keeping `trig=4/4` and even gaining one
onset — speech is captured measurably sooner with no detection cost. 300ms
is the cliff (drops a recording, `min_onsets` 0). Pre-roll is the
complementary lever: it leaves the onset *count* untouched (10 across all
three values) while pulling `onset1` earlier by ~250ms per 256ms of
pre-roll — the recovered-soft-attack effect iter-191 measured, now visible
as a one-number corpus aggregate instead of a per-recording table. The two
levers stack: a 100ms debounce + 256ms pre-roll would pull the opening
earlier on both axes. This is the replay-validated case for lowering the
client `debounceMs` *default* below 200 (the knob has shipped since
iter-196) once the busier corpus confirms it holds.

### Worst-case onset ceiling (iter-198)

The `onset1` mean above answers "does timing move earlier on average?" but
hides a single recording captured far too late — exactly the failure
`min_onsets` exists to catch on the *count* axis. iter-198 adds the
companion `onset1_max` column: the latest first onset across detected
recordings, a ceiling a sweep wants to *minimize*. The same two sweeps,
now showing both:

```
python fixtures/replay_vad.py --sweep debounce_ms --sweep-values 100,200,300
```
| debounce_ms | trig | onset1 (mean) | onset1_max (worst) |
|-------------|------|---------------|--------------------|
| **100** | 4/4 | **1271.3ms** | 2995.4ms |
| 200 (default) | 4/4 | 1532.5ms | 2995.4ms |
| 300 | **3/4** | 2190.4ms | **3343.7ms** |

```
python fixtures/replay_vad.py --sweep preroll_ms --sweep-values 0,256,512
```
| preroll_ms | trig | onset1 (mean) | onset1_max (worst) |
|------------|------|---------------|--------------------|
| 0 (default) | 4/4 | 1532.5ms | 2995.4ms |
| 256 | 4/4 | 1282.9ms | 2740.0ms |
| 512 | 4/4 | 1091.3ms | 2484.5ms |

**What the ceiling reveals that the mean does not:** lowering the debounce
200→100ms pulls the *mean* ~261ms earlier yet leaves `onset1_max` pinned at
**2995ms** — the worst recording isn't debounce-limited, so the debounce
knob can't help its slow opening (one recording simply has a late real
speech start). Pre-roll, by contrast, moves *both* numbers in lockstep
(it shifts the emitted onset of every detected recording uniformly), so it
pulls the ceiling down too (2995→2484ms at 512ms pre-roll). That is the
practical takeaway for backlog item 5: a smaller debounce improves the
*typical* opening but does nothing for the worst case, whereas pre-roll is
the lever that tightens the tail — they are complementary, not redundant.

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
   would recover clipped utterance openings. _iter-191: modelled in the
   replay harness (`VadParams.preroll_ms` / `--preroll-ms`). On the seed
   corpus a 256–512ms pre-roll pulls every recording's first `onset_ms`
   earlier by ~250–510ms (see the table above) with no change to onset count
   and no segment overlap — validated by `TestPrerollRecoversOpening` in
   `tests/integration/test_vad_recordings.py`. **iter-193: wired into the
   client.** `ContinuousListener` now takes a `prerollMs` option (default `0`
   = exact historical parity) and keeps a bounded ring of pre-onset frames
   (`_pushPreroll` / `_drainPreroll`, sized from `prerollMs` × sample rate).
   While listening, every sub-threshold frame — and the frames of any broken
   speech candidate — fold into the ring; at the commit point in
   `_handleFrame` the ring is drained and prepended to `this.chunks`, so the
   committed segment recovers the clipped soft attack. Covered by the repo's
   first JS suite, `client/voice-capture.test.js` (`node --test`): parity at
   `prerollMs=0`, prepend-on-commit, ring capacity bound, candidate-fold,
   drain-reset, sample-rate scaling, `stop()` clear._
3. **[threshold] Confirm 0.006 holds as the corpus grows.** The regression
   test guards against silent regressions. Re-run `replay_vad.py --sweep
   threshold ...` each lap on any newly-synced recordings; if a new
   recording misses at 0.006, investigate (gain stage? far-field? noise
   floor?) before lowering further. _iter-190: swept 0.004→0.020 on the
   seed corpus — the detection cliff is between 0.010 and 0.015, so 0.006
   sits with comfortable margin (`min_onsets=1`, `trig=4/4`)._
4. **[gain] Software gain experiment.** _iter-190: swept gain 1.0→3.0 at
   threshold 0.006. Onsets and speaking frames rise monotonically; even at
   2.0× the lifted silence floor (~0.0006) stays ~3× under the gate, so a
   1.5–2.0× gain recovers more speech without false triggers. iter-192:
   `--grid threshold,gain` over the corpus (see the grid table above) shows
   the joint effect a single-axis sweep can't — at threshold 0.015 the
   far-field recording misses at gain 1.0 but **1.5× gain recovers it**
   (`trig` 3/4 → 4/4). So a future gain stage would let the threshold rise
   back toward 0.010–0.015 without losing far-field speech. Detection is
   monotone along both axes, pinned by `TestGridSweep`. **iter-195: wired
   into the client.** `ContinuousListener` now takes a `gain` option
   (default `1.0` = exact unity-gain no-op; non-finite/non-positive falls
   back to unity). `_handleFrame` pre-amplifies each frame via `_applyGain`
   before RMS detection, the raw callback, segment storage, and the pre-roll
   ring — equivalent to a `gainNode` sitting upstream, and a faithful mirror
   of the replay harness `frame_rms(samples * gain)` model. Amplified audio
   also flows into the committed WAV so STT sees the louder signal. The JS
   suite (`client/voice-capture.test.js`, +8 tests) covers unity-gain parity
   + zero-copy, the unity fallback for bad values, per-sample scaling, the
   sub-threshold→over-threshold far-field recovery (0.01 frame: misses at 1×,
   commits at 2×), segment/raw-callback/pre-roll amplification. The chosen
   client operating point stays gain 1.0 (the grid shows unity already gives
   `trig=4/4` on the seed corpus); the knob is the documented hedge if a
   busier corpus ever needs a stricter threshold._
5. **[debounce] Tune onset debounce against clipping.** _iter-190: swept
   100→300ms at threshold 0.006. 100–200ms all keep `trig=4/4`; 300ms drops
   a recording. 100ms recovers slightly more speaking frames (less clipped
   opening)._ **iter-196: the client `ContinuousListener` now accepts a
   `debounceMs` option (default 200 = exact historical parity), mirroring the
   replay harness `debounce_ms` field — previously the onset debounce was a
   hard-coded `200` literal in `_handleFrame`. The knob is validated, non-finite/
   negative values fall back to 200, and `0` commits on the first surviving
   candidate frame._ **iter-197: onset *timing* validated.** The sweep now
   reports `mean_first_onset_ms` (`onset1`), so the timing claim is measurable
   in one pass: on the seed corpus 100ms debounce pulls the mean first onset
   **~261ms earlier** (1532.5→1271.3ms) while keeping `trig=4/4` and gaining an
   onset (see the onset-timing section above); 300ms is the cliff (drops a
   recording). The replay evidence now supports lowering the client `debounceMs`
   *default* to 100; the remaining gate is confirming it on a busier corpus
   before changing the shipped default._
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
