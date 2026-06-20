# TTS-pacing WPM mirror (living doc)

A data-driven log of the geno-voice **TTS-pacing** system — the offline
simulator and on-device calibration behind the WPM mirror, the feature that
matches the agent's speaking rate to the user's. Seeded by iter-320, after the
simulator/calibrator surface had grown five laps (iters 215–319) without a
prose home.

The live pacing path lives in `session/wpm_mirror.py`. The agent watches the
user's words-per-minute (the iter-046 `bot_wpm` machinery, applied to the user's
transcript) and nudges the Kokoro `speed` multiplier toward a target derived
from `base_wpm` — a hardware/voice calibration constant (`DEFAULT_BASE_WPM`,
165.0). A user who speeds up pulls the agent faster; a user who slows down pulls
it slower. Two knobs shape the dynamics:

- **`strength`** (`DEFAULT_STRENGTH`, 0.5) — the damping. `0` ignores the user
  (the mirror is off); `1` snaps to the user's implied speed every turn. In
  between, the agent eases toward the target over several turns.
- **The intelligibility band** (`min_speed`/`max_speed`, 0.8/1.3) and the
  **deadband** (`min_delta`, 0.05) — the mirror never leaves
  `[min_speed, max_speed]` no matter how extreme the user, and a per-turn change
  smaller than `min_delta` is dropped so the rate doesn't churn.

The engine is pure stdlib (no audio, no torch), so the whole subsystem is
unit-testable offline. Two `gv` subcommands expose it:
`simulate-mirror` replays a user-WPM arc through the mirror, and
`calibrate-base-wpm` backs `base_wpm` out of real TTS renders.

## `gv simulate-mirror` — replay a user-WPM arc (iter-218)

The offline twin of the live `SpeedController` fold: feed it a per-turn
user-WPM arc and it reports how the mirror's `speed` multiplier evolves, with
**no audio and no live session**. The headless analogue of talking to the agent
at varying paces and watching the rate track you.

```
gv simulate-mirror --wpms 120,140,200,140,120        # human-readable trajectory
gv simulate-mirror --wpms 120,140,200,140,120 --json # machine-readable
gv simulate-mirror --wpms 120,140,200,140,120 --csv  # flat per-turn speed curve
```

A `<= 0` entry in `--wpms` models a **silent / no-measurement turn** (the user
said nothing the agent could clock), and the mirror holds the prior speed
through it. `--initial-speed` sets the rate before turn 1 (default `1.0`),
`--base-wpm` the convergence calibration (default `165.0`), and `--strength` the
damping (default `0.5`).

The trajectory report names the **ideal target** (`user_wpm / base_wpm`,
clamped to the band), the **final gap** (residual to that target — a small gap
means the arc converged), the **max step** (the largest single-turn lurch — a
big step means a jumpy, unnatural ride), and the per-turn speed curve. The
`--json` payload carries the same scalars plus a `turns` array
(`turn`/`user_wpm`/`speed`); the `--csv` writes one `turn,user_wpm,speed` row
per turn for plotting. `--json` and `--csv` are mutually exclusive (argparse
rejects passing both).

### `--grid` — sweep base_wpm × strength (iter-217)

`base_wpm` and `strength` interact: a higher `base_wpm` lowers the target speed
for the same user pace, so the `strength` that converges smoothly shifts with
it. A single-axis sweep is too coarse to pick the joint operating point. `--grid`
replays the arc once per cell of the two axes and scores every cell:

```
gv simulate-mirror --grid --wpms 120,140,200,140,120
gv simulate-mirror --grid --wpms 120,140,200,140,120 \
    --base-wpms 150,165,180 --strengths 0.3,0.5,0.7 --json
gv simulate-mirror --grid --wpms 120,140,200,140,120 --csv
```

`--base-wpms` / `--strengths` are the two axes (comma-separated; defaults
`150,165,180` × `0.3,0.5,0.7`). Each cell's **score** blends the `|final_gap|`
(did it converge?) and the `max_step` lurch (was the ride smooth?); the report
prints the data-driven **best pick** — the lowest-score cell. The `--json`
payload carries a flat `cells` list plus a `best` object; the `--csv` adds an
`is_best` column (`1`/`0`) so the pick survives the flattening to a spreadsheet.

**`--lurch-weight` (iter-318)** tunes the score's trade-off: it weights the
lurch term relative to the convergence term (default `0.5`). Raise it to favor a
smoother approach over converging exactly; lower it to favor hitting the target
even at the cost of a jumpier ride. It threads into both the picker and all
three renderers, so the displayed `score` always reflects the weight the best
pick was decided on. Trajectory mode has no score, so the knob is inert there.

```
gv simulate-mirror --grid --wpms 120,140,200,140,120 --lurch-weight 2.0
```

**`--min-speed` / `--max-speed` / `--min-delta` (iter-319)** override the
intelligibility band and deadband for the run. In `--grid` mode the band becomes
the template every cell clones (so a sweep can run against a *non-seed* band —
e.g. a wider window for a faster voice); in trajectory mode it lands directly on
the single config. They apply to **both** modes. `--max-speed` must be
`>= --min-speed`; a bad pair is caught by `WpmMirrorConfig` and reported as a
clean `error:` line rather than a traceback.

```
gv simulate-mirror --wpms 60,60,400,400 --min-speed 0.5 --max-speed 2.0
gv simulate-mirror --grid --wpms 120,200,120 --min-speed 0.7 --max-speed 1.5 --min-delta 0
```

## `gv calibrate-base-wpm` — measure base_wpm from renders (iter-220)

`base_wpm` is **not tunable by replay** (iter-219's hard finding): it is the
bot's actual words-per-minute at Kokoro `speed=1.0` — a hardware/voice
calibration — and the simulator's own `ideal = user_wpm / base_wpm` *uses* it to
define the target, so sweeping it scores each cell against its own moving target
(circular). The right `base_wpm` for a deployment is therefore a **measurement**:
synthesize a known-length script, time the audio, and back out the rate the
voice clocks at speed 1.0.

This subcommand is the audio-free arithmetic core of that calibration. Each
`--samples` triple is one render — `words:audio_seconds[:speed]` — and the
handler folds them into a robust **median** `implied_base_wpm` plus spread
(min↔max) and drift-vs-nominal diagnostics:

```
gv calibrate-base-wpm --samples 50:18.2 50:9.1:2.0          # human-readable
gv calibrate-base-wpm --samples 50:18.2 48:17.9 52:18.6 --json
gv calibrate-base-wpm --samples 50:18.2 48:17.9 52:18.6 --csv
```

Each sample derives the measured `bot_wpm` (the iter-046 `words·60/audio_seconds`)
and the `implied_base_wpm` (that rate normalized back to speed 1.0, i.e.
`bot_wpm / speed`). The `speed` field defaults to the `1.0` calibration point.
`--nominal` sets the `base_wpm` to report **drift** against (default `165.0`):
positive drift means the voice is faster than nominal. A large **spread** means
the renders disagree — don't trust the median. The `--json` payload nests a
`samples` list and a `calibration` object; the `--csv` writes one row per sample
with the aggregate trailing as `#` comment lines. `--json` and `--csv` are
mutually exclusive.

### `--verdict` — adopt or keep (iter-222/223)

The raw numbers leave the decision to the operator. `--verdict` folds them into
an explicit **re-seed vs keep-nominal** call, gated on three thresholds:

```
gv calibrate-base-wpm --samples 50:18.2 48:17.9 52:18.6 --verdict
```

- **`--spread-max`** (default `10.0`) — if the renders disagree by more than this
  the median isn't trustworthy, so keep nominal.
- **`--drift-min`** (default `5.0`) — drift smaller than this is absorbed by the
  damped mirror, so re-seeding isn't worth it.
- **`--min-samples`** (default `3`) — fewer samples than this isn't a robust
  median.

The verdict prints the decision (`re-seed to X` / `keep the current nominal`),
the reason, and the gates it checked. It is **human prose, not a data record**,
so `--json` and `--csv` suppress it — a programmatic consumer scripts the
re-seed call off the `drift` field directly.

## The format trio

Every analysis surface here carries the full **human / `--json` / `--csv`**
trio, matching the VAD-analysis surfaces (`gv vad` / `vad-diff` / `vad-sweep` /
`vad-grid`, documented in *Voice-capture tuning*). The human report is the
default; `--json` is the nested machine surface for programmatic consumers;
`--csv` is the flat surface for spreadsheets and plots. The two machine surfaces
are derived views of the same numbers — they never disagree with the human
report, only re-shape it.

## Methodology notes

- The engine (`session/wpm_mirror.py`) is **pure stdlib** — no audio, no torch,
  no clock. The `gv` handlers load it lazily by file path so the parser stays
  importable on any host (including the x86_64 Linux loop runner with no
  pipecat / Kokoro). Tests drive the renderers and handlers directly with an
  injected `log`, so no real I/O happens.
- The simulator is the *offline twin* of the live `SpeedController` fold, not a
  reimplementation: it shares the `WpmMirrorConfig` and `mirrored_speed` the
  live path uses, so a tuning answer found offline holds online.
- `base_wpm` calibration ships only the arithmetic core. The real Kokoro render
  that *produces* the `words:audio_seconds` samples is the on-device follow-on
  (it gates on a real synth); the operator runs that, then feeds the durations
  here.
