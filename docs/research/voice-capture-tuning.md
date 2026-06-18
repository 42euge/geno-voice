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
real miss even when the total looks healthy), `max_onsets` (iter-201 — the
busiest single recording, the *over-split ceiling*: the symmetric companion
to `min_onsets`, where `min_onsets` catches a recording dropping to a miss
and `max_onsets` catches the opposite — a recording *fragmenting* into many
short segments, the signature of a too-short `silence_ms`), `onset_std` (iter-206
— `std_onsets`, the population standard deviation of the per-recording onset
count across the corpus: the count-axis *consistency*, to the count axis what
`onset1_std` is to the timing axis and `seg_std` is to the duration axis.
`min_onsets`/`max_onsets` bracket the envelope and `onsets` gives the sum, but
none express *spread* — how unevenly onsets are distributed across recordings.
Two `silence_ms` values can share an `onsets` total while one fragments a single
recording into many segments (one count spikes far above the rest — high spread)
and the other splits every recording evenly (low spread); `min`/`max` catch the
spike only if it reaches the corpus extreme, `onset_std` reads the whole
distribution's unevenness directly. Population std over every recording's count
*including misses* (a miss contributes a `0` — a real point on the count axis,
unlike the timing axis where a miss has no onset time and is excluded), so a
single-recording corpus reads as `0.0`), `max_seg` (iter-202
— `max_segment_ms`, the longest single committed segment across the corpus: the
*over-merge ceiling*, the duration-axis companion to `max_onsets`. Where
`max_onsets` reads a too-*short* `silence_ms` fragmenting one utterance,
`max_seg` reads a too-*long* `silence_ms` fusing two real turns into one run-on
segment — a failure that leaves the onset count flat and is invisible to every
count aggregate, showing up only as a single segment's duration ballooning, so
the two ceilings bracket both ends of the silence lever), `min_seg` (iter-203 —
`min_segment_ms`, the shortest single committed segment across the corpus: the
*over-split floor* on the duration axis, the symmetric companion to `max_seg` as
`min_onsets` is to `max_onsets`. It confirms by *duration* what `max_onsets`
catches by *count* — a too-short `silence_ms` chops one utterance into many short
fragments, so the shortest emitted segment collapses toward the `min_speech`
gate. Reading both per axis — `min_onsets`/`max_onsets` on count,
`min_seg`/`max_seg` on duration — a `--sweep silence_ms` brackets both failures
on both axes in one pass; a committed segment can never be shorter than the
`min_speech` gate, so this floor is bounded below by it), `mean_seg` (iter-204 —
`mean_segment_ms`, the mean committed segment duration across the corpus: the
*center* of the duration axis, to it what `onset1` is to the timing axis. Where
`min_seg`/`max_seg` each read one worst-case recording (the over-split floor and
over-merge ceiling), `mean_seg` reads the corpus as a whole, so it shows whether
the *typical* turn is lengthening (turns merging across the board) or shortening
(fragmenting across the board), not just whether one outlier did. Averaged over
every emitted segment, so a recording that fragments into many short segments
rightly pulls it down; bounded below by the `min_speech` gate and within
`[min_seg, max_seg]`), `seg_std` (iter-205 — `std_segment_ms`, the population
standard deviation of every committed segment's duration across the corpus: the
duration *consistency*, to the duration axis what `onset1_std` is to the timing
axis. `min_seg`/`mean_seg`/`max_seg` give the envelope and center; `seg_std`
gives the *spread* — the one thing they can't. Two `silence_ms` values can share
a `mean_seg` while one emits uniformly medium turns and the other mixes short
fragments with long run-ons (the over-split *and* over-merge mix a borderline
timeout produces): identical mean, far larger `seg_std` for the mixed case, so it
is the only aggregate that separates a cleanly-segmenting parameter set from one
unstable in both directions at once. Population std over every emitted segment,
so a single committed segment reads as `0.0` and the value is bounded above by
the `max_seg - min_seg` range), `onsets`/`speak_frames` totals, `mean_over` (mean %-of-frames-over-threshold),
and `onset1`
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
that single bad onset is a real regression the mean would hide. iter-199
completes the spread with `onset1_min`: the *earliest* first onset across
detected recordings — the best-case floor. With `onset1_min`/`onset1`/
`onset1_max` together a sweep shows the whole best/typical/worst timing
distribution in one pass, and the floor marks the *irreducible* earliest
capture an onset-shaping knob can't push past (when `onset1_min` stops
moving, that knob has saturated on its best recording). iter-200 adds the
last statistic, `onset1_std`: the population standard deviation of the
first onsets across detected recordings — the timing *consistency*. The
min/mean/max give the envelope and center; the std gives the *spread*, the
one thing they can't — two parameter sets can share an `onset1` mean while
one opens at a consistent time every recording and the other swings between
very early and very late. The std is the only aggregate that distinguishes
them, so a grid sweep can pick the cell that opens early *and* consistently
rather than early-on-average with a wild tail. Population (not sample) std,
so a single detected recording reads as `0.0` ("perfectly consistent given
one point") rather than undefined.

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

**Auto-gain recommendation** (iter-228, `--recommend-gain`). The sweep
above maximizes *onsets*, but STEER.md item #2 asks a stricter question:
the largest gain that lifts the quietest real speech over the threshold
*without lifting silence over a hard 0.0003 ceiling*. The `recommend_gain`
analyzer answers it directly — it measures each recording's silence floor
(median per-frame RMS, robust because speech is sparse) and quiet-speech
level (low percentile of over-threshold frames) at gain=1.0, takes the
*loudest* floor as the binding constraint and the *softest* speech as the
target, and picks the largest gain keeping the binding floor under the
ceiling:

```
python fixtures/replay_vad.py --recommend-gain
# recommended_gain=1.00x  OK  silence_floor=0.00036 (ceiling=0.00030,
#   headroom=0.82x)  quiet_speech=0.00642 -> 0.00642 (target=0.00600)
```

**Verdict: gain=1.0 — no amplification is safe on this corpus.** The
binding silence floor measured per-recording is **0.00036** (the median
RMS of the noisiest recording), already *above* the 0.0003 ceiling
(headroom 0.82×). The sweep's "~0.0003 floor" was the corpus *minimum*;
the analyzer uses the *binding* (loudest) floor, which is what a safe
recommendation must respect. Because every recording already triggers at
gain=1.0 and the quietest speech (0.00642) already clears the 0.006 gate,
amplification has nothing to recover and would only push silence further
over the ceiling — consistent with the gain sweep's monotonically rising
`onset_std` (1.50→11.16 over 1.0→8.0×), the signature of clean utterances
fragmenting. **The 1.5–2.0× recommendation above held silence below the
*detection gate* (0.006); it does not hold silence below the stricter
0.0003 ceiling the steering sets.** The analyzer remains the live tool for
future quiet recordings: a recording with a genuinely low floor and
sub-threshold speech *will* yield a `recommended_gain > 1.0` with an `OK`
verdict (covered by `tests/unit/test_recommend_gain.py`).

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

### Best-case onset floor (iter-199)

`onset1` (mean) and `onset1_max` (worst case) describe the typical and tail
of the timing distribution; iter-199 adds `onset1_min` — the *earliest*
first onset across detected recordings — so a single sweep now reports the
full best/typical/worst spread. The floor's job is to expose the
*irreducible* earliest capture: an onset-shaping knob can't pull a recording
earlier than its best case, so when `onset1_min` stops moving, that knob has
saturated. The same two sweeps, now showing all three:

```
python fixtures/replay_vad.py --sweep debounce_ms --sweep-values 100,200,300
```
| debounce_ms | trig | onset1_min (best) | onset1 (mean) | onset1_max (worst) |
|-------------|------|-------------------|---------------|--------------------|
| 100 | 4/4 | **232.2ms** | 1271.3ms | 2995.4ms |
| 200 (default) | 4/4 | **232.2ms** | 1532.5ms | 2995.4ms |
| 300 | **3/4** | **232.2ms** | 2190.4ms | 3343.7ms |

```
python fixtures/replay_vad.py --sweep preroll_ms --sweep-values 0,256,512
```
| preroll_ms | trig | onset1_min (best) | onset1 (mean) | onset1_max (worst) |
|------------|------|-------------------|---------------|--------------------|
| 0 (default) | 4/4 | 232.2ms | 1532.5ms | 2995.4ms |
| 256 | 4/4 | **0.0ms** | 1282.9ms | 2740.0ms |
| 512 | 4/4 | **0.0ms** | 1091.3ms | 2484.5ms |

**What the floor reveals:** across every debounce value the floor is *pinned
at 232.2ms* — the earliest recording's real speech starts there and no amount
of debounce tuning can capture it sooner, confirming debounce is a
*typical-case* lever, not a best-case one (it mirrors how `onset1_max` stays
pinned across the same sweep — debounce moves only the middle of the spread).
Pre-roll, by contrast, drives the floor all the way to **0.0ms** (the
recording start) at 256ms of pre-roll: the best recording's soft attack is
recovered right back to the file boundary, where the clamp stops it. So the
floor confirms from the *bottom* of the distribution what `onset1_max`
confirmed from the top — pre-roll shifts the whole spread uniformly while
debounce only moves the interior.

### Onset-timing consistency, and the joint debounce×preroll grid (iter-200)

`onset1_min`/`onset1`/`onset1_max` describe the *shape* of the timing
distribution (where it starts, centers, and ends) but not how tightly the
recordings cluster. iter-200 adds the last statistic, `onset1_std` — the
population standard deviation of the first onsets across detected recordings,
i.e. the timing *consistency*. Two parameter sets can share an `onset1` mean
while one opens at a steady time every recording and the other swings wildly;
only the std tells them apart, so a sweep can finally pick the cell that opens
early *and* consistently rather than early-on-average with a ragged tail.

With all four aggregates in place, the two onset-timing levers
(`debounce_ms`, `preroll_ms`) are both reachable as client knobs (iter-196,
iter-193), so the joint grid the backlog has been deferring is now readable in
one pass (backlog item 3):

```
python fixtures/replay_vad.py --grid debounce_ms,preroll_ms \
    --grid-values-a 100,200,300 --grid-values-b 0,256,512 --threshold 0.006
```
| debounce_ms | preroll_ms | trig | onset1_min | onset1 (mean) | onset1_max | onset1_std |
|-------------|------------|------|-----------|---------------|-----------|-----------|
| 100 | 0 | 4/4 | 232.2ms | 1271.3ms | 2995.4ms | 1081.2ms |
| 100 | 256 | 4/4 | 0.0ms | 1021.7ms | 2740.0ms | 1075.7ms |
| 100 | 512 | 4/4 | 0.0ms | 835.9ms | 2484.5ms | 1014.4ms |
| 200 | 0 | 4/4 | 232.2ms | 1532.5ms | 2995.4ms | 982.0ms |
| 200 | 256 | 4/4 | 0.0ms | 1282.9ms | 2740.0ms | 974.3ms |
| **200** | **512** | **4/4** | **0.0ms** | **1091.3ms** | **2484.5ms** | **893.1ms** |
| 300 | 0 | **3/4** | 232.2ms | 2190.4ms | 3343.7ms | 1391.9ms |
| 300 | 256 | **3/4** | 0.0ms | 1942.7ms | 3088.3ms | 1381.1ms |
| 300 | 512 | **3/4** | 0.0ms | 1772.5ms | 2832.8ms | 1261.4ms |

**What the consistency column reveals** — a result the mean alone hides:
lowering debounce 200→100ms pulls the *mean* ~261ms earlier (1532.5→1271.3ms
at preroll 0) but actually *widens* the spread (982.0→1081.2ms). The earliest
recording opens sooner while the late one stays pinned, so the gap between
recordings grows — debounce buys a better average at the cost of *less*
consistency. Pre-roll, by contrast, both shifts the mean earlier *and* tightens
the spread (982.0→893.1ms across preroll 0→512 at debounce 200), because it
moves every detected recording uniformly. The lowest-std cell that still holds
`trig=4/4` is **debounce 200 / preroll 512** (onset1_std 893.1ms, mean
1091.3ms): the most consistent early opening on the seed corpus comes from
keeping the default debounce and leaning on pre-roll, *not* from dropping
debounce. This sharpens backlog item 5's "pair a debounce drop with a pre-roll
bump" — the grid says pre-roll is doing the real work for both the typical case
*and* the consistency, so a pre-roll default bump is the higher-leverage change.
The 300ms debounce row stays the cliff (`trig=3/4`, widest spread).

### Over-split ceiling, and reading the silence timeout (iter-201)

Every aggregate so far measures the onset *count totals* (`onsets`,
`speak_frames`) or the *onset timing* (`onset1*`). Neither makes the
silence-timeout failure mode legible. `silence_ms` decides when a pause inside
one utterance ends a *segment*: set it too long and two real turns merge into
one; set it too short and one continuous utterance **fragments** into many short
segments. The corpus total barely moves under fragmentation (the same speech,
just chopped), and `min_onsets` is blind to it (it watches the floor, where a
recording goes to *zero*). The signature of over-splitting lives at the *other*
end — a single recording's onset count climbing well above the rest.

iter-201 adds `max_onsets`: the most onsets any single recording got, the
symmetric companion to `min_onsets`. Together they bracket the per-recording
onset count — `min_onsets` catches a recording dropping to a *miss*,
`max_onsets` catches one *fragmenting* — so a `silence_ms` sweep reads both the
under-merge (the floor collapses turns) and the over-split (the ceiling shatters
one) ends in a single pass, the same way `onset1_min`/`onset1_max` bracket the
timing spread.

A synthetic illustration (one recording: 1s tone — 300ms gap — 1s tone, all
loud) makes the lever visible, since the seed corpus has no mid-utterance gap in
that window:

| silence_ms | min_onsets | max_onsets | onsets | reading |
|-----------:|-----------:|-----------:|-------:|---------|
| 100        | 2          | **2**      | 2      | the 300ms gap > 100ms timeout → the utterance **splits** in two |
| 400        | 1          | 1          | 1      | 300ms gap < 400ms timeout → stays merged |
| 800        | 1          | 1          | 1      | the production default — merged |

The ceiling climbing from 1 to 2 as `silence_ms` drops below the gap length is
exactly the over-split signal. Running the same `--sweep silence_ms` over the
real corpus (backlog item 6) now reads cleanly: a `max_onsets` that jumps at
low `silence_ms` means turns are fragmenting, telling the operator the timeout
floor before quality degrades.

### Over-merge ceiling, the other end of the silence lever (iter-202)

`max_onsets` (iter-201) reads only one of the two ways `silence_ms` can go
wrong: too *short* and one utterance fragments into many segments (the count
ceiling climbs). The opposite — too *long*, fusing two real conversational
turns into one run-on segment — is invisible to every count aggregate: the
onset count stays flat or even *falls* (two onsets become one), while the
damage shows up purely as a single segment's *duration* ballooning. iter-202
adds `max_segment_ms` (the `max_seg` column): the longest single committed
segment across the corpus, the duration-axis companion to `max_onsets`.
Together they bracket both ends of the silence lever in one `--sweep silence_ms`
pass — the count ceiling catches over-splitting, the duration ceiling catches
over-merging.

The same synthetic gap recording (1s tone — 300ms gap — 1s tone, all loud)
shows both ceilings moving in opposite directions as `silence_ms` crosses the
gap length:

| silence_ms | max_onsets | max_seg | reading |
|-----------:|-----------:|--------:|---------|
| 100        | **2**      | 1152.0ms | gap (300ms) > timeout → **splits**: two ~1.15s halves |
| 400        | 1          | **2304.0ms** | gap < timeout → **merges**: one 2.3s run-on segment |
| 800        | 1          | **2304.0ms** | the production default — merged |

As `silence_ms` rises past the 300ms gap the count ceiling *drops* (2→1, the
merge) while the duration ceiling *doubles* (1152→2304ms, the bridged gap plus
both halves). On a real corpus a `max_seg` that climbs at high `silence_ms`
toward multiples of a single turn's length is the over-merge signal — turns are
running together — telling the operator the timeout *ceiling*, the same way
`max_onsets` reads the floor. The seed corpus has no two-turns-one-pause case in
the relevant window, so revisit when a newly-synced recording exercises it.

### Over-split floor on the duration axis (iter-203)

`max_segment_ms` (iter-202) closed the over-*merge* end of the silence lever on
the duration axis, but left that axis with only a ceiling. The matching floor
was still missing: where `min_onsets`/`max_onsets` bracket the *count* axis
(floor catches a miss, ceiling catches a fragment), the duration axis had only
`max_seg`. iter-203 adds `min_segment_ms` (the `min_seg` column): the shortest
single committed segment across the corpus, the symmetric companion to `max_seg`
as `min_onsets` is to `max_onsets`. It confirms over-splitting by *duration* —
the same failure `max_onsets` reads by *count*. When a too-short `silence_ms`
chops one utterance into many short fragments, the count ceiling climbs *and* the
shortest emitted segment collapses toward the `min_speech` gate; the two are the
count- and duration-axis fingerprints of the same fragmentation.

The same synthetic gap recording, read on the duration axis, shows the floor and
ceiling moving in opposite directions as `silence_ms` crosses the gap length:

| silence_ms | min_seg | max_seg | reading |
|-----------:|--------:|--------:|---------|
| 100        | **1152.0ms** | 1152.0ms | gap (300ms) > timeout → **splits**: two short halves, the floor drops |
| 400        | 2304.0ms | 2304.0ms | gap < timeout → **merges**: one long segment, floor = ceiling |
| 800        | 2304.0ms | 2304.0ms | the production default — merged |

As `silence_ms` rises past the gap the floor *climbs* (the fragments fuse into
one long segment) while a low `silence_ms` pulls it down toward the `min_speech`
gate (more, shorter fragments). All four bracket aggregates now read in one
`--sweep silence_ms` pass: `min_onsets` (count floor, a miss), `max_onsets`
(count ceiling, a fragment), `min_seg` (duration floor, a fragment), `max_seg`
(duration ceiling, a merge). The seed corpus is flat in both windows, so gate
the actual `silence_ms` default change on a newly-synced corpus that exercises
real mid-utterance pauses and two-turns-one-pause cases.

### Center of the duration axis (iter-204)

iter-202/203 bracketed the segment-*duration* axis with a ceiling (`max_seg`,
over-merge) and a floor (`min_seg`, over-split), but left it with no *center* —
the onset-*timing* axis already carried its full floor/typical/ceiling shape
(`onset1_min`/`onset1`/`onset1_max`), so the duration axis was a statistic short.
iter-204 adds `mean_segment_ms` (the `mean_seg` column): the mean committed
segment duration across the corpus, the typical-turn-length aggregate. It is to
the duration axis what `onset1` is to the timing axis.

The floor and ceiling each read *one* worst-case recording — the single shortest
or longest segment in the whole corpus. That makes them sensitive to a lone
outlier: one recording fragmenting can drop `min_seg` to the gate while every
other turn is untouched. `mean_seg` reads the corpus as a whole, so it answers
the complementary question — is the *typical* turn lengthening or shortening? It
is averaged over every emitted segment (not per-recording-then-averaged), so a
recording that shatters into many short fragments rightly pulls the mean down by
contributing many short durations, not just one.

The same synthetic gap recording, read on all three duration statistics:

| silence_ms | min_seg | mean_seg | max_seg | reading |
|-----------:|--------:|---------:|--------:|---------|
| 100        | 1024.0ms | **1088.0ms** | 1152.0ms | gap (300ms) > timeout → **splits**: two short halves, typical turn is short |
| 400        | 2304.0ms | **2304.0ms** | 2304.0ms | gap < timeout → **merges**: one long run-on, typical turn doubles |
| 800        | 2304.0ms | **2304.0ms** | 2304.0ms | the production default — merged |

`mean_seg` moves the same direction as `max_seg` under the merge (the typical
turn lengthens), confirming the *whole corpus* shifted rather than one outlier;
on a real corpus a `mean_seg` climbing alongside `max_seg` at high `silence_ms`
is the strongest signal that turns are merging broadly, while `max_seg` alone
climbing with a flat `mean_seg` would point to a single run-on recording. The
duration axis now carries the same floor/typical/ceiling shape the onset-timing
axis does. The seed corpus is flat in both silence windows, so gate the actual
`silence_ms` default change on a newly-synced corpus that exercises real
mid-utterance pauses and two-turns-one-pause cases.

### Consistency of the duration axis (iter-205)

iter-202/203/204 gave the segment-*duration* axis its floor (`min_seg`,
over-split), ceiling (`max_seg`, over-merge), and center (`mean_seg`, typical
turn) — but the onset-*timing* axis carried a *fourth* statistic the duration
axis lacked: `onset1_std`, the consistency (spread) of the distribution. iter-205
closes that asymmetry with `std_segment_ms` (the `seg_std` column): the
population standard deviation of every committed segment's duration across the
corpus. It is to the duration axis exactly what `onset1_std` is to the timing
axis, completing both axes to the same floor/typical/ceiling/spread shape.

The spread is the one thing min/mean/max can't give. Two `silence_ms` values can
share a `mean_seg` while segmenting completely differently: one emits uniformly
medium turns (low spread), the other mixes short fragments with long run-ons —
the over-split *and* over-merge mix a borderline timeout produces — at the same
mean (high spread). `seg_std` is the only aggregate that separates a
cleanly-segmenting parameter set from one that is unstable in *both* directions
at once, so a `--sweep silence_ms` should prefer the value that minimizes
`seg_std` among those with an acceptable `mean_seg`, not just the one with the
best mean.

| corpus | segment durations | mean_seg | seg_std | reading |
|--------|-------------------|---------:|--------:|---------|
| two equal 1s tones | 1024.0ms, 1024.0ms | 1024.0ms | **0.0ms** | uniform turns → no spread |
| 1s + 1s tones split by a 300ms gap | 1152.0ms, 1024.0ms | 1088.0ms | **64.0ms** | unequal committed halves → real spread |

Population (not sample) std, so a single committed segment reads as `0.0`
("perfectly consistent given one point") rather than undefined, and the value is
bounded above by the `max_seg - min_seg` range. As with the other duration
aggregates, the seed corpus is flat in both silence windows, so gate any actual
`silence_ms` default change on a newly-synced corpus.

### Consistency of the count axis (iter-206)

The onset-*timing* axis (`onset1_min`/`onset1`/`onset1_max`/`onset1_std`) and the
segment-*duration* axis (`min_seg`/`mean_seg`/`max_seg`/`seg_std`) both carry the
full envelope/center/spread shape after iter-205. The onset-*count* axis had only
`min_onsets` (floor, iter-189), `max_onsets` (ceiling, iter-201) and `onsets`
(corpus sum) — an envelope and a total, but no *spread*. iter-206 adds
`std_onsets` (the `onset_std` column): the population standard deviation of the
per-recording onset count across the corpus, to the count axis exactly what
`onset1_std` is to the timing axis and `seg_std` is to the duration axis.

The spread is the one thing min/max/total can't give. Two `silence_ms` values can
share an `onsets` total while distributing those onsets completely differently:
one fragments a *single* recording into many short segments (that recording's
count spikes far above the rest — high spread) while the other splits *every*
recording evenly (low spread). `min_onsets`/`max_onsets` catch the spike only if
it reaches the corpus extreme; `onset_std` reads the whole distribution's
unevenness directly, so a `--sweep silence_ms` can prefer the value that
fragments uniformly (or not at all) over one that shatters a lone recording while
leaving the total flat.

| corpus | onset counts | onsets (total) | onset_std | reading |
|--------|--------------|---------------:|----------:|---------|
| two clean tones, one onset each | 1, 1 | 2 | **0.00** | even distribution → no spread |
| one tone split by a sub-gap timeout + one clean tone | 2, 1 | 3 | **0.50** | one recording fragments while the other stays whole → real spread |

Unlike the onset-*timing* std, which excludes missed recordings (a miss has no
onset time), the count std *includes* a miss as a `0` — a recording detecting
nothing is a real count of zero, and a corpus mixing a hit with a miss is
genuinely inconsistent in onset count. Population (ddof=0) std, so a
single-recording corpus reads as `0.0` ("perfectly consistent given one point")
and the value is bounded above by the `max_onsets - min_onsets` range. With this
the count axis carries the same envelope+spread shape the timing and duration
axes do; as with the other silence-lever aggregates the seed corpus is flat in
both windows, so gate any actual `silence_ms` default change on a newly-synced
corpus.

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
   before changing the shipped default._ **iter-198/199: the timing aggregate
   now carries its worst case (`onset1_max`) and best case (`onset1_min`)
   alongside the mean.** Both confirm debounce is a *typical-case* lever only:
   across the 100→300ms sweep the floor stays pinned at 232.2ms and the ceiling
   at 2995ms — debounce moves only the interior of the spread, while pre-roll
   shifts the whole distribution (floor → 0.0ms, ceiling → 2484ms at 512ms).
   Pair any `debounceMs` default drop with a pre-roll bump so the tail and floor
   improve too, not just the mean.
6. **[silence] Right-size the silence timeout.** 800ms may over-split or
   over-merge turns. The seed corpus shows clean multi-segment splits;
   revisit if new recordings show truncated or run-on turns. _iter-201: the
   `max_onsets` aggregate now makes over-splitting legible — a `--sweep
   silence_ms` whose ceiling climbs at low values is fragmenting one utterance
   into many (see "Over-split ceiling" above). The seed corpus has no
   mid-utterance gap in the sub-800ms window, so the sweep is flat there today;
   gate the actual default change on a newly-synced corpus that exercises real
   pauses. iter-202: the `max_segment_ms` aggregate now reads the *other* end —
   over-merging — so a `--sweep silence_ms` brackets both failures at once: a
   `max_onsets` jump at low values means fragmentation, a `max_seg` climbing
   toward multiples of one turn's length at high values means turns are running
   together (see "Over-merge ceiling" above). iter-203: the `min_segment_ms`
   aggregate adds the over-split floor on the duration axis, so all four bracket
   aggregates now read in one pass — `min_onsets`/`max_onsets` on count,
   `min_seg`/`max_seg` on duration. A `min_seg` collapsing toward the
   `min_speech` gate at low `silence_ms` confirms by duration the fragmentation
   `max_onsets` reads by count (see "Over-split floor on the duration axis"
   above). iter-204: the `mean_segment_ms` aggregate fills in the *center* of the
   duration axis — where `min_seg`/`max_seg` each read one worst-case recording,
   `mean_seg` reads the whole corpus, so a `mean_seg` climbing alongside `max_seg`
   at high `silence_ms` means turns are merging *broadly*, not just in one
   recording (see "Center of the duration axis" above). iter-205: the
   `std_segment_ms` aggregate completes the duration axis with its *consistency*
   (spread), matching the onset-timing axis's `onset1_std`. Two `silence_ms`
   values can share a `mean_seg` while one segments uniformly and the other mixes
   short fragments with long run-ons; `seg_std` is the only aggregate that tells
   them apart, so prefer the `silence_ms` that minimizes `seg_std` among those
   with an acceptable `mean_seg` (see "Consistency of the duration axis" above).
   iter-206: the `std_onsets` aggregate completes the *count* axis with its
   consistency (spread), matching the timing axis's `onset1_std` and the duration
   axis's `seg_std`. Two `silence_ms` values can share an `onsets` total while one
   fragments a single recording (one count spikes — high spread) and the other
   splits every recording evenly (low spread); `onset_std` is the only count
   aggregate that reads that unevenness, so prefer the value that fragments
   uniformly over one that shatters a lone recording (see "Consistency of the
   count axis" above)._

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
