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

## Silero neural VAD — the primary path (iter-231)

**Energy-RMS VAD is a dead end for continuous speech.** The ground-truth
recording `voice-20260618-110355.wav` (31s of continuous speaking) proves it:
its in-speech noise floor is ~0.016 RMS and the speech median is only ~0.023 —
too close. NO threshold × silence combination (swept 0.006–0.015 × 250–800ms,
plus hysteresis) breaks it into more than **1** segment, so the utterance never
closes ("VAD triggered but wouldn't untrigger"). Worse, the count is not even
monotonic in threshold: on `voice-20260617-161615.wav` the *lower* 0.006 gate
keeps the inter-utterance gap above threshold and **merges** two turns into one
segment, while 0.015 splits them — the merging-vs-splitting failure that makes
energy thresholds unworkable on real rooms.

**Silero VAD (neural) is the fix.** It scores P(speech) per 32ms window with a
small pretrained model, distinguishing speech from room-tone *regardless of the
energy floor*. The live mic path already uses it
(`pipecat_server.py`: `SileroVADAnalyzer(params=VADParams(min_volume=0.01,
stop_secs=0.8))`); iter-231 brings the same model to the `:5111` server and to a
headless replay harness so the recording corpus — not a mic — is the proof.

The segmenter lives in `vad/silero.py` (lazy-loaded, dependency-degrading) and
is replayed over the whole corpus by `fixtures/replay_silero.py`:

```
python fixtures/replay_silero.py                 # segments per recording
python fixtures/replay_silero.py --compare       # Silero vs energy-VAD counts
python fixtures/replay_silero.py --json          # machine-readable
```

### `gv vad` — single-file CLI segmentation (iter-233)

For ad-hoc inspection of one WAV (no server, no whole-corpus replay), the gv
CLI exposes the same batch segmenter:

```
gv vad recording.wav                             # speech regions for one WAV
gv vad recording.wav --threshold 0.7             # stricter P(speech) gate
gv vad recording.wav --min-silence-ms 500        # shorter end-of-turn hangover
gv vad recording.wav --json                      # machine-readable (iter-234)
```

The defaults track `SileroParams` (`threshold=0.5`, `min_speech_ms=250`,
`min_silence_ms=800` = the pipecat `stop_secs=0.8`, `speech_pad_ms=30`,
`max_speech_s=inf`); `--max-speech-s none` (or `inf`/`off`) never force-splits.
The report prints the sample rate, duration, segment count, total speech, and
the per-region `start–end (duration)` table. When `silero-vad` is not installed
the command prints a one-line install hint and exits cleanly (the same
degrade-don't-die contract as the server's 503). `cmd_vad` takes injected
`segmenter`/`availability`/`log` seams so `tests/unit/test_gv_vad.py` covers it
without torch; `tests/integration/test_gv_vad_cli.py` runs it over the real
corpus and re-pins the 31s ≥2-segment gate through the CLI.

**`--json` (iter-234)** swaps the human report for a single JSON document so the
segmentation can feed scripts and tooling, mirroring
`fixtures/replay_silero.py --json` / `SileroResult.to_dict`. The payload carries
the same keys (`name`, `sample_rate`, `duration_s`, `num_segments`, `speech_s`,
`segments[]` of `start_s`/`end_s`/`duration_s`, all rounded to 3 places) plus an
`available` flag and the echoed `threshold`. On a host without `silero-vad` the
JSON is `{"available": false, "hint": …}` so a consumer detects the degraded
path from the document itself rather than parsing prose.

### `gv vad-diff` — compare two thresholds (iter-235)

The first consumer of the `gv vad --json` surface. Tuning the P(speech) gate
against the corpus previously meant running `gv vad` twice and eyeballing the
two reports. `gv vad-diff` runs the segmenter twice over one WAV — once per
threshold, all other knobs shared — and reports the signed delta:

```
gv vad-diff recording.wav                                 # 0.5 vs 0.7 (defaults)
gv vad-diff recording.wav --threshold-a 0.3 --threshold-b 0.9  # wider sweep
gv vad-diff recording.wav --json                          # machine-readable delta
```

The human report names the file, both thresholds, and the `A → B` segment-count
and speech-seconds transitions with explicit signs, e.g.
`segments: 5 → 4 (-1)` / `speech total: 16.2s → 15.2s (-1.0s)`. A higher gate is
typically a *subset* of a lower one (fewer regions, less speech), so the deltas
are usually `≤ 0`. The `--json` payload carries `threshold_a`/`threshold_b`,
both sides (`num_segments_a/b`, `speech_s_a/b`), and the signed
`num_segments_delta`/`speech_s_delta`, so a sweep harness can consume it
directly. The same degrade-to-`{"available": false}` contract as `gv vad`
applies. `cmd_vad_diff` reuses the iter-233 injected-dependency seams;
`vad_segmentation_delta` is the pure delta core, tested without torch in
`tests/unit/test_gv_vad.py`, and `tests/integration/test_gv_vad_cli.py` proves
the diff equals two independent `gv vad --json` runs over the real corpus.

### `gv vad-sweep` — tabulate N values of one knob (iter-236)

`gv vad-diff` compares the P(speech) gate at exactly two points; `gv vad-sweep`
generalises that to a sweep over N values of one knob (the single-file analogue
of `fixtures/replay_silero.py`'s sweep over the corpus), so the knob's *elbow* —
where recovered speech falls off as the gate tightens, or where regions merge as
the hangover lengthens — is visible at a glance rather than as a single A-vs-B
delta. The default swept knob is the P(speech) gate (`--thresholds`); iter-238
adds the trailing-silence hangover as a second axis (`--min-silences`, below),
iter-239 adds the minimum-speech floor as a third (`--min-speeches`, below),
iter-253 adds the symmetric region padding as a fourth (`--speech-pads`, below),
and iter-256 adds the force-split ceiling as a fifth (`--max-speeches`, below —
the only axis measured in seconds, not ms):

```
gv vad-sweep recording.wav                                  # 0.3,0.5,0.7,0.9 (defaults)
gv vad-sweep recording.wav --thresholds 0.1,0.3,0.5,0.7,0.9 # custom gates
gv vad-sweep recording.wav --json                           # machine-readable rows
gv vad-sweep recording.wav --csv                            # flat CSV for plots (iter-237)
gv vad-sweep recording.wav --target 3                       # data-driven best-value pick (iter-244)
gv vad-sweep recording.wav --target 3-5                     # tolerance band: 0 distance inside (iter-246)
gv vad-sweep recording.wav --target 3-                      # open band: at least 3 regions (iter-247)
gv vad-sweep recording.wav --target -5                      # open band: at most 5 regions (iter-247)
gv vad-sweep recording.wav --target 3,5,7                   # set: 3 OR 5 OR 7 regions (iter-248)
gv vad-sweep recording.wav --target 3>5>7                   # preference: prefer 3, accept 5, then 7 (iter-249)
gv vad-sweep recording.wav --target 3,5:2                   # weighted: prefer 3, accept 5 but 2 worse (iter-250)
gv vad-sweep recording.wav --target 3,5:1.5                 # fractional weight: 5 is 1.5 segments worse (iter-251)
gv vad-sweep recording.wav --target 3,5*1.5                 # scaled: prefer 3, accept 5, drift past costs 1.5x (iter-252)
```

The human report is a small table — the WAV name, a `threshold / segments /
speech` column header, then one row per threshold:

```
silero VAD sweep — voice-20260618-110355.wav
  threshold  segments  speech
       0.30         5   17.3s
       0.50         5   16.2s
       0.70         4   15.5s
       0.90         4   15.2s
```

Because a stricter gate can only keep or shrink recovered speech, reading down
an ascending-threshold sweep the segment count / speech total are non-increasing
(an integration test pins this monotonicity over the corpus's hardest
recording). The `--thresholds` list is parsed by `unit_interval_list_type`
(comma-separated gates in `[0, 1]`, order preserved, empty list rejected) and
all other knobs are shared across runs. The `--json` payload carries
`{"available": true, "name", "sweep": [{"threshold", "num_segments",
"speech_s"}, …]}`, so a plotting/tuning script can consume it directly. The same
degrade-to-`{"available": false}` contract as `gv vad` applies. `cmd_vad_sweep`
reuses the iter-233 injected-dependency seams; `vad_segmentation_sweep` is the
pure core, tested without torch in `tests/unit/test_gv_vad.py`, and
`tests/integration/test_gv_vad_cli.py` proves each sweep row equals an
independent `gv vad --json` run over the real corpus.

**`--csv` (iter-237)** emits the same rows as `--json` but as a flat
`threshold,num_segments,speech_s` grid that pipes straight into a spreadsheet
or plotting script (`pandas.read_csv`, gnuplot, `numpy.loadtxt`) without a
JSON-parsing step:

```
threshold,num_segments,speech_s
0.3,5,17.3
0.5,5,16.2
0.7,4,15.5
0.9,4,15.2
```

`--csv` and `--json` are mutually exclusive (argparse rejects passing both).
The CSV body is a pure data grid — the WAV name is *not* a column (it would only
repeat per row), unlike the human table's title line and the JSON `name` key.
When `silero-vad` is absent the output is a single `# silero VAD unavailable: …`
comment line so a degraded run stays self-describing. An integration test pins
that the `--csv` rows describe the same segmentation as `--json` over the real
corpus. The 1-D `--csv` and `--json` surfaces are *axis-agnostic* — each
stringifies whichever dimension the run sweeps — so the CSV↔JSON agreement holds
on every axis, not just the default `threshold`. iter-268 unit tests pin that a
multi-row `min_speech_ms` and a multi-row `speech_pad_ms` sweep parse from the
CSV `DictReader` back to the *exact* `--json` `sweep` payload (the first CSV
column keyed by the swept ms-axis name), and iter-269 completes the trio with
the `min_silence_ms` axis (the most common non-default sweep — it tunes the
trailing-silence gate that ends one utterance and starts the next). Together
they close the gap left by the threshold-only cross-surface round-trip and the
single-row ms-axis header tests: the round-trip is now proven on `threshold`
(default), `max_speech_s` (seconds), and all three ms axes. iter-270 lifts the
same cross-surface proof to the 2-D **grid**: `render_vad_grid_csv` and
`render_vad_grid_json` are likewise axis-agnostic, so a multi-cell grid on a
fully non-default axis pair (`min_speech_ms` rows × `speech_pad_ms` columns) now
parses from the CSV `DictReader` back to the *exact* `--json` `grid` payload,
cell for cell in row-major order. (The prior grid round-trip test compared the
CSV only against the shared `vad_segmentation_grid` data layer on the default
`threshold × min_silence_ms` axes — it never round-tripped the two machine
surfaces directly, nor on a non-default axis pair.) iter-271 extends that grid
twin to the `inf` no-cap baseline: the never-force-split sentinel takes two
different textual forms across the surfaces — the CSV writes the bare token
`inf` while the JSON relies on Python emitting `Infinity` (which `json.loads`
reads back as `float('inf')`) — so a grid with `max_speech_s` rows `[inf, 5]`
crossed with `speech_pad_ms` columns now proves both surfaces recover the
sentinel as `float('inf')` in the *same* cells, with no `Infinity` spelling
leaking into the CSV body. iter-272 mirrors that inf grid twin with the
sentinel on the **column** axis (iter-271 placed it on the row axis): a
`threshold` rows × `max_speech_s` columns grid with `[inf, 5]` columns rides
the inf baseline in the second CSV column of *every* row, and both surfaces
still recover it as `float('inf')` in the same row-major cells — closing the
cross-surface gap left by the iter-267 col-axis inf test, which round-tripped
the CSV only against the shared data layer, never directly against the JSON
emitter. iter-273 closes the last open cross-surface seam — the `--target`
**pick**. When a target is set the JSON twin grows a `best` cell (and a `top`
shortlist), but the CSV stays a pure data grid with no pick columns, so a CSV
consumer must *re-derive* the operator's pick by parsing the flat table back to
cells and re-running `pick_best_grid_cell` / `pick_top_grid_cells`. iter-273
pins that agreement: the CSV-derived best equals the JSON-embedded `best`
(distance stripped), the re-derived distance matches, and the top shortlist
agrees cell-for-cell — including the row-major order of two equidistant
runners-up — so the pick is identical no matter which surface a tuning script
reads. iter-274 extends that pick agreement to the *non-default*
`tie_break="speech"`: the JSON twin threads the tie-break into
`pick_best_grid_cell` / `pick_top_grid_cells`, and a CSV consumer re-running the
same pickers with the same `"speech"` tie-break recovers the same speech-broken
pick. The fixture is built so the two tie-breaks genuinely *disagree* (equal
distance, different recovered speech → row-major and speech name different
cells), so the test would catch a JSON path that silently dropped the tie-break
and fell back to row-major — a divergence iter-273's default-tie-break fixture
could not see. iter-275 generalises the pick agreement past the *scalar* target
to an iter-246 closed `(lo, hi)` **tolerance band**: every count inside the
inclusive window scores distance 0, so a band makes multiple cells tie at the
band floor where a scalar would separate them. The JSON twin forwards the band
opaquely to `grid_cell_distance`, and a CSV consumer re-running the pickers with
the same band recovers the same in-band pick and band-scored shortlist. The
fixture is built so a scalar at the band's lower edge would pick a *different*
cell (the lone exact hit) than the band (whose floor ties two cells, broken
row-major), so the test would catch a JSON path that collapsed the band tuple to
a scalar. iter-276 carries the pick agreement to the first form whose precedence
lives at the *sort-key* layer rather than the distance: an iter-249
`{"prefer": […]}` **preference** target. `grid_cell_distance` treats a preference
identically to a flat set (the min over its elements), and only
`grid_cell_sort_key` inserts a `_preference_rank` secondary key so that, among
cells tied at equal distance, the one nearest a *more-preferred* element wins. A
CSV consumer re-running the pickers with the same `{"prefer": …}` dict recovers
the same preference-broken pick. The fixture ties two cells at distance 0 and
adds a flat-`set` control of the same two elements: the set falls back to the
row-major tie (the earlier cell), while the preference flips the pick toward the
more-preferred element — so the test catches a JSON path that flattened the
`{"prefer": …}` dict to a plain set and dropped the preference rank key. iter-277
carries the pick agreement to the first form whose preference is folded back
*into* the distance rather than living at the sort-key layer: an iter-250
`{"weighted": […]}` target, where each element scores its raw `|Δ|` *plus* its
penalty and the set takes the min over those penalised distances. Unlike a
preference (which only breaks exact-distance ties), a weight can *override* a
raw-distance gap — a less-preferred-but-closer cell can lose to a
preferred-but-farther one. The fixture lands one cell exactly on the
penalty-bearing accepted element (raw distance 0, penalised 2) and another at raw
distance 1 from the penalty-free preferred element (penalised 1), so the penalty
flips the pick; a flat-`set` control of the same two elements carries no
penalties, lets the exact hit win, and picks a *different* cell — so the test
catches a JSON path that dropped the penalties and collapsed the weighted set to
a plain set of its elements. iter-278 carries the same proof to the
*multiplicative* twin: an iter-252 `{"scaled": […]}` target, where each element's
cost is raw `|Δ|` *times* its factor. The distinction from the additive weight is
that a factor *grows* with distance — a far cell on a high-factor element loses
harder the farther it drifts, while an exact hit stays free on any factor
(`0 × factor = 0`). The fixture scales the preferred element by `×3` so a cell at
raw distance 1 from it (scaled 3) loses to a cell at raw distance 2 from the
`×1` accepted element (scaled 2); a flat-`set` control of the same two elements
carries no factors, picks the closer cell, and lands a *different* winner — and
because the additive and multiplicative folds choose different cells for the same
element set, the iter-277 weighted fixture cannot catch a JSON path that confused
the two. iter-279 carries the cross-surface pick agreement to an iter-247 *open*
band — a `(lo, None)` ("at least `lo`") or `(None, hi)` ("at most `hi`") target,
where one edge is unbounded so the open side simply skips its bound check. This is
a distinct distance shape from the iter-275 *closed* band: the in-band region is
one-sided and unbounded, so arbitrarily many cells tie at the floor. The fixture
targets `(5, None)`, ties a count-12 and a count-7 cell at distance 0, and a
*closed* `(5, 9)` control pushes count 12 back out (distance 3) to pick a different
cell — so the test catches a JSON path that coerced the open `None` edge to a
finite bound or collapsed the open band to a scalar at `lo`. iter-280 closes the
last documented form: the iter-248 flat **set** — a plain list of elements scored
by the *min* distance to any one of them. The earlier dict-form fixtures only ever
used a flat set as a *control* to prove their preference/weight/factor diverged
from it, so the set itself was never the pinned cross-surface subject. The fixture
targets `[3, 8]`, lands an exact hit on element `8` (distance 0), and a *scalar*
control equal to the set's first element `3` picks a different, nearer-to-3 cell —
so the test catches a JSON path that kept only the first element (collapsing the
set to a scalar) or coerced the list to a closed band. With this, every
`grid_cell_distance` target form (scalar, closed band, open band, set, preference,
weighted, scaled) is cross-surface pinned.

**`--min-silences` — a second sweep axis (iter-238).** The default axis is the
P(speech) gate (`--thresholds`); passing `--min-silences` instead sweeps the
*trailing-silence hangover* `min_silence_ms` — how much quiet must follow speech
before a region ends. The gate is then held fixed at the scalar `--threshold`
(default `0.5`). The two axes are mutually exclusive (argparse rejects passing
`--thresholds` together with `--min-silences`); exactly one knob varies per run.

```
gv vad-sweep recording.wav --min-silences 200,400,800,1600           # sweep the hangover
gv vad-sweep recording.wav --min-silences 200,400,800 --threshold 0.7 # hold the gate at 0.7
gv vad-sweep recording.wav --min-silences 200,400,800 --csv          # flat CSV for plots
```

The column label, the JSON/CSV first column, and the `--json` `axis` key all
become `min_silence` / `min_silence_ms` so the swept dimension is readable
straight off the data. Hangover values print as bare integers in the human
table (`400`, not `0.40`). Over `voice-20260618-110355.wav`:

```
silero VAD sweep — voice-20260618-110355.wav
  min_silence  segments  speech
        200         9   14.5s
        400         5   15.7s
        800         5   16.2s
       1600         5   16.2s
```

A *longer* hangover can only merge adjacent regions (never split them), so the
segment count is non-increasing as the value rises — the mirror of the
threshold axis's speech-non-increasing property. Here the elbow is sharp: at
200 ms the recording fragments into 9 regions (silences between clauses end
segments early), settling to 5 once the hangover reaches ~400 ms. The
`--min-silences` list is parsed by `nonneg_float_list_type` (comma-separated
durations `≥ 0`, order preserved, empty list rejected; `0` is legitimate). The
`--json` payload gains an `"axis"` key (`"threshold"` or `"min_silence_ms"`) and
keys each row by that name. Integration tests pin that each silence-sweep row
equals an independent `gv vad --json` run at that hangover, that segments are
monotone non-increasing across rising hangovers, and that `--csv` agrees with
`--json` on this axis — all over the real corpus.

**`--min-speeches` — a third sweep axis (iter-239).** Passing `--min-speeches`
instead sweeps the *minimum-speech floor* `min_speech_ms` — the shortest speech
region the segmenter keeps; anything briefer is dropped as noise. As with
`--min-silences`, the gate is held fixed at the scalar `--threshold` (default
`0.5`). All four axes are mutually exclusive (argparse rejects passing more than
one); exactly one knob varies per run.

```
gv vad-sweep recording.wav --min-speeches 50,100,200,400,800        # sweep the floor
gv vad-sweep recording.wav --min-speeches 50,100,200 --threshold 0.7 # hold the gate at 0.7
gv vad-sweep recording.wav --min-speeches 50,100,200 --csv          # flat CSV for plots
```

The column label, the JSON/CSV first column, and the `--json` `axis` key all
become `min_speech` / `min_speech_ms`. Floor values print as bare integers in
the human table (`400`, not `0.40`), sharing the millisecond formatter with the
hangover axis. Over `voice-20260618-110355.wav`:

```
silero VAD sweep — voice-20260618-110355.wav
  min_speech  segments  speech
         50         5   16.2s
        100         5   16.2s
        200         5   16.2s
        400         4   15.7s
        800         2   14.3s
```

A *higher* floor can only drop short regions (never add them), so the segment
count is non-increasing as the value rises — the same monotonicity shape as the
hangover axis, from the opposite cause (culling, not merging). Here the elbow is
at ~400 ms: below it every region clears the floor (5 segments), then short
clauses start getting culled, collapsing to 2 by 800 ms. The `--min-speeches`
list is parsed by the same `nonneg_float_list_type` validator as `--min-silences`
(comma-separated durations `≥ 0`, order preserved, empty list rejected; `0` is
legitimate). The `--json` `"axis"` key now also takes `"min_speech_ms"` and keys
each row by that name. Integration tests pin that each speech-sweep row equals an
independent `gv vad --json` run at that floor, that segments are monotone
non-increasing across rising floors, and that `--csv` agrees with `--json` on
this axis — all over the real corpus.

**`--speech-pads` — a fourth sweep axis (iter-253).** Passing `--speech-pads`
instead sweeps the *symmetric region padding* `speech_pad_ms` — the margin Silero
adds to each end of every recovered region. Too little clips the talker's onsets
and tails (a word's leading consonant or trailing fricative lands outside the
region); too much pads regions until adjacent ones touch and merge. As with the
other ms axes, the gate is held fixed at the scalar `--threshold` (default `0.5`),
and the four axes are mutually exclusive.

```
gv vad-sweep recording.wav --speech-pads 0,20,40,60,100            # sweep the padding
gv vad-sweep recording.wav --speech-pads 0,20,40 --threshold 0.7   # hold the gate at 0.7
gv vad-sweep recording.wav --speech-pads 0,20,40 --csv             # flat CSV for plots
```

The column label, the JSON/CSV first column, and the `--json` `axis` key all
become `speech_pad` / `speech_pad_ms`. Pad values print as bare integers in the
human table (`40`, not `0.04`), sharing the millisecond formatter with the
hangover and floor axes. Unlike the floor and hangover, the segment count is
*not* monotone in padding: more padding can only merge regions (never split
them), so the count is non-increasing — but the *speech seconds* it recovers
rises as padding stops clipping onsets/tails, then plateaus once regions begin to
merge. The elbow is the smallest pad that recovers the talker's edges without yet
fusing distinct utterances. The `--speech-pads` list is parsed by the same
`nonneg_float_list_type` validator as the other ms axes (comma-separated durations
`≥ 0`, order preserved, empty list rejected; `0` is legitimate — the unpadded
boundaries). The scalar `--speech-pad-ms` is held fixed when sweeping the other
axes and ignored while `--speech-pads` sweeps.

**`--max-speeches` — a fifth sweep axis (iter-256).** Passing `--max-speeches`
instead sweeps the *force-split ceiling* `max_speech_s` — the maximum length
Silero lets a single region grow before it is force-split into multiple
segments. A loose cap (or the `inf` no-cap baseline) leaves a long monologue as
one region; tightening the cap chops it into progressively more segments. This
is the **only sweep axis measured in seconds**, not milliseconds: the gate is
still held fixed at the scalar `--threshold` (default `0.5`), and all five axes
are mutually exclusive.

```
gv vad-sweep recording.wav --max-speeches 5,10,20,inf             # sweep the ceiling (inf = no cap)
gv vad-sweep recording.wav --max-speeches 5,10,20 --threshold 0.7 # hold the gate at 0.7
gv vad-sweep recording.wav --max-speeches 5,10,20 --csv           # flat CSV for plots
```

The column label, the JSON/CSV first column, and the `--json` `axis` key all
become `max_speech` / `max_speech_s`. Because this is the seconds axis, values
print compactly via `%g` in the human table (`5`, `12.5`, and the no-cap
sentinel as `inf`) — not the bare-integer ms formatter the other four axes
share. The segment count is non-decreasing as the cap *tightens* (a smaller cap
can only split regions, never merge them), so the elbow is the largest cap that
still keeps the longest natural utterance intact before the ceiling starts
chopping it. The `--max-speeches` list is parsed by the dedicated
`max_speech_list_type` validator (the seconds twin of `nonneg_float_list_type`):
each comma-separated token runs through the scalar `max_speech_type`, so the
`inf`/`none`/`off` "never split" sentinels and the positive-only rule (a `0`s cap
would force-split forever) carry through per element, and `inf` can anchor the
no-cap baseline mid-sweep. The scalar `--max-speech-s` is held fixed when
sweeping the other axes and ignored while `--max-speeches` sweeps. This axis is
shared with `gv vad-grid`'s `--max-speeches` column (iter-255), so every grid
column axis is now also a 1-D sweep axis.

#### `--target` / `--top` / `--tie-break` — a data-driven best-value pick (iter-244)

`gv vad-grid` gained a data-driven best-cell pick across iter-241→243; iter-244
brings the **same machinery to the 1-D sweep**, closing the sweep↔grid feature
gap. A sweep row carries the same `num_segments` / `speech_s` keys a grid cell
does, so the very same pickers (`pick_best_grid_cell`, `pick_top_grid_cells`,
`grid_cell_distance`, `grid_cell_sort_key`) drive the sweep pick unchanged — no
parallel implementation.

```bash
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3            # best swept value
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3 --top 3    # ranked shortlist
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3 --tie-break speech
gv vad-sweep recording.wav --min-silences 400,800 --threshold 0.7 --target 3  # works on any swept axis
```

`--target N` (the segment count you expect) surfaces a trailing `best:` line
naming the swept value whose recovered segment count is closest to `N`, scored by
`|num_segments - N|`. `--top K` lists the K closest values as a ranked shortlist
(nearest first), its head always the `best:` value. `--tie-break {row-major,speech}`
breaks equal-distance ties: `row-major` (the default) keeps the earlier swept
value (output unchanged byte-for-byte from iter-236); `speech` prefers the value
that recovered the most speech (clips the talker least). Over
`voice-20260618-110355.wav` with `--target 3 --top 3`:

```
silero VAD sweep — voice-20260618-110355.wav
  threshold  segments  speech
       0.30         5   17.3s
       0.50         5   16.2s
       0.70         4   15.5s
       0.90         4   15.2s
  best: threshold=0.70 (4 segments, |Δ|=1 from target 3)
  top 3 (closest to target 3):
    1. threshold=0.70  4 segments  |Δ|=1
    2. threshold=0.90  4 segments  |Δ|=1
    3. threshold=0.30  5 segments  |Δ|=2
```

The semantics match `gv vad-grid`'s `--target`/`--top`/`--tie-break` exactly (see
above): distance is always the primary key, `--top` rides along with `--target`,
and all three are derived views — `--json` adds `target` / `tie_break` / `best` /
`top` keys (`best`/`top` cells augmented with a `distance` key), but the flat
`--csv` data grid ignores them. Without `--target` the output is byte-for-byte the
iter-236 shape — no pick keys leak in. Integration tests pin, over the real
corpus, that the `best` minimises `|num_segments - target|` over the same sweep the
run tabulated, the `top` head equals `best`, and the absent-target payload keeps
the iter-236 shape.

#### `--target LO-HI` — a tolerance band (iter-246)

A single `--target N` scores against one exact segment count, but operators
often want a *window* — "anywhere from 3 to 5 regions is fine, just not 1 and
not 8". iter-246 lets `--target` take a `LO-HI` range (both for `vad-sweep` and
`vad-grid`); any count **inside** the inclusive band scores distance `0` (every
in-band count is equally perfect), and a count outside scores the gap to the
nearest edge (below `LO` → `LO - count`; above `HI` → `count - HI`).

```bash
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3-5          # band, not a point
gv vad-grid  recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800 --target 3-5
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3-5 --top 3  # band + shortlist
```

The band reuses the existing `grid_cell_distance` scoring, so `--top`,
`--tie-break`, and the `--json` payload all flow through unchanged: a banded
distance is just another lower-is-better key. The `best:` / `top N:` lines and
the `--json` `target` render the band as `3-5` (the JSON `target` is a `[3, 5]`
array); the scalar form (`--target 3`) is byte-for-byte the iter-241→245
behaviour, since a scalar still parses to a bare `int` and a degenerate band
`(n, n)` reduces to the scalar distance to `n`. Edges are non-negative whole
numbers and `LO <= HI` (an inverted band is rejected as a typo). A `--target`
with no `-` is the scalar form, exactly as before.

#### `--target N-` / `--target -N` — open-ended bands (iter-247)

A closed `LO-HI` band caps both ends, but often only one end matters — "at
least 3 regions, however many more is fine" or "at most 5, the fewer the
better". iter-247 lets either edge be **empty**: `--target 3-` means "at least
3" (`(3, None)` — no upper bound) and `--target -5` means "at most 5"
(`(None, 5)` — no lower bound). The open side simply skips its bound check, so
any count on the satisfied side scores distance `0` and only the closed edge can
produce a non-zero gap.

```bash
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3-   # at least 3 regions
gv vad-grid  recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800 --target -5  # at most 5
```

The `best:` / `top N:` lines render an open band exactly as typed (`3-`, `-5`),
and the `--json` `target` carries the open edge as `null` (`[3, null]`,
`[null, 5]`). Everything else — `--top`, `--tie-break`, the pickers — flows
through the same `grid_cell_distance` machinery, since an open band is still a
`(lo, hi)` tuple (with `None` marking the open edge). Note `--target -1` now
parses as the open band "at most 1", not a (rejected) negative count — a bare
negative segment count is no longer expressible, which is harmless since nobody
targets a negative count.

#### `--target A,B,C` — a set of acceptable counts (iter-248)

A band (closed or open) accepts a contiguous window; but sometimes the
acceptable counts are **disjoint** — "3 OR 5 segments, but nothing between" (two
phrasings that segment cleanly, the in-between count being an artefact). iter-248
lets `--target` take a comma-separated SET: `--target 3,5,7` means "3 OR 5 OR 7".
Each element is itself a scalar or a band, so they compose — `--target 3,5-7`
means "3 OR anywhere from 5 to 7". The distance is the **minimum** over the
elements, so a count satisfying ANY listed target scores `0` and otherwise scores
the gap to the nearest one.

```bash
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3,5,7   # 3 OR 5 OR 7 regions
gv vad-grid  recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800 --target 3,5-7  # 3 OR a 5-7 band
```

The `best:` / `top N:` lines render a set comma-joined exactly as typed
(`3,5,7`), and the `--json` `target` serialises as a JSON array of its elements
(a band element nests as its own `[lo, hi]` array — `3,5-7` → `[3, [5, 7]]`).
Everything else — `--top`, `--tie-break`, the pickers — flows through the same
`grid_cell_distance` machinery, which recurses as a min-over-elements. A set is
deduped preserving first-seen order, and a single-element set (`3,3`) collapses
to the bare element so scalar/band output stays byte-for-byte unchanged. An empty
element (`3,`, `3,,5`) is rejected as a typo.

#### `--target A>B>C` — a preference order (iter-249)

A SET treats every listed count as equally acceptable; but an operator often
**prefers** one count yet would **settle** for another — "prefer 3 regions, but 5
is fine, and 7 only as a last resort". iter-249 lets `--target` take a
`>`-separated PREFERENCE order: `--target 3>5>7` accepts ANY listed count (the
distance is the same min-over-elements as a set, so all listed counts score `0`),
but UNLIKE a set it carries a precedence so a distance **tie** breaks toward the
**earlier-listed** (more-preferred) count. Each element is itself a scalar or a
band, so they compose — `--target 3>5-7` means "prefer 3, else anywhere from 5 to
7".

```bash
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3>5>7   # prefer 3, accept 5, then 7
gv vad-grid  recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800 --target 3>5-7  # prefer 3, else a 5-7 band
```

The preference is the **first** tie-break — stronger intent than grid position or
recovered speech (`--tie-break speech`), so among cells equally close to the
target the one nearest a more-preferred count wins; `--tie-break` only decides
cells that ALSO tie on preference rank. The `best:` / `top N:` lines render a
preference `>`-joined exactly as typed (`3>5>7`), and the `--json` `target`
serialises as a `{"prefer": [...]}` object carrying the listed order (a band
element nests as its own `[lo, hi]` array — `3>5-7` → `{"prefer": [3, [5, 7]]}`).
A preference is deduped preserving first-seen order, and a single-element
preference (`3>3`) collapses to the bare element so scalar/band output stays
byte-for-byte unchanged. An empty element (`3>`, `>5`) is rejected as a typo, and
mixing `,` (set) with `>` (preference) in one target is rejected — they are
different composition operators.

#### `--target A,B:W` — a weighted set (iter-250)

A `>` preference breaks only an **exact** distance tie toward the earlier count.
But an operator may want the preferred count to win even when it is slightly
**farther** — "I'll take 3 segments at distance 1 over 8 at distance 0, because 8
is over-segmenting". iter-250 lets a comma-set element carry a `:penalty` weight:
`--target 3,8:2` means "prefer 3, accept 8 **but treat it as 2 segments worse**
than it actually is". The penalty is **added** to that element's distance, so the
weight folds preference INTO the distance — unlike a `>` preference, it can
**override a raw-distance gap**, not just break a tie. An element with no `:`
carries penalty `0` (the iter-248 set element, unweighted); each element is itself
a scalar or band, so they compose (`3,5-7:2`).

```bash
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3,8:2   # prefer 3, accept 8 but 2 worse
gv vad-grid  recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800 --target 3,5-7:2  # weighted band
```

The distance is the **min** over each element's `(raw distance + penalty)`, so a
count routes through whichever element is cheapest — a count near an unweighted
neighbour never pays a distant element's penalty. The `best:` / `top N:` lines
render the set comma-joined with each non-zero penalty appended (`3,8:2`), the
`|Δ|` shown being the **penalised** distance, and the `--json` `target`
serialises as a `{"weighted": [[element, penalty], ...]}` object (a band element
nests as its own `[lo, hi]` array — `3,5-7:2` → `{"weighted": [[3, 0], [[5, 7],
2]]}`). A weighted set is deduped on the element preserving first-seen order (the
first penalty wins), and one that collapses to a single element drops the
now-useless penalty and reduces to the bare element (a lone penalty is a constant
offset that cannot change any pick). A `:` weight **requires** a `,` set (it is
meaningless on a single element) and cannot be combined with `>` (preference) —
both express preference, so stacking them is rejected as ambiguous. Each penalty
is a non-negative number.

**Fractional weights (iter-251).** The penalty may be **fractional** —
`--target 3,5:1.5` means "the accepted count 5 is 1.5 segments worse than the
preferred 3". Because the penalty is additive, a whole-number weight can only
*step* the "preferred count wins at a larger raw distance" boundary across
integers; a fractional weight lands it **between** them. With `3,6:1.5`, count 4
(penalised 1.0) beats the exact-accepted count 6 (penalised 1.5) which still beats
count 5 (penalised 2.0) — an ordering neither a penalty of 1 nor 2 can place. An
**integral** float collapses back to an int (`5:2.0` → `5:2`), so every
integer-penalty result — parse value, rendered line, `|Δ|`, and `--json` `target`
— is byte-for-byte the iter-250 output; only a genuinely fractional weight stays a
float. Negative, NaN, and infinite penalties are rejected (a negative weight would
make a count *better* than its raw distance, which the other element's penalty
already expresses; an infinite one is a degenerate "never pick this").

```bash
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3,5:1.5  # fractional weight: 5 is 1.5 worse
```

#### `--target A,B*F` — a scaled set (iter-252)

The `:penalty` weight is **additive** — a fixed offset on an element's distance,
so a less-preferred count is "N segments worse" no matter how far the cell drifts,
and the penalty bites even an **exact** hit (`0 + penalty`). An operator who thinks
**proportionally** ("count 5 is acceptable, but every segment I drift *past* it
should hurt more") has no expression for that. iter-252 adds the `*factor`
**multiplicative** twin: `--target 3,5*1.5` means "prefer 3, accept 5 **but drift
past it costs 1.5×**". The factor **multiplies** that element's distance
(`distance * factor`), so — unlike the additive penalty — an exact hit stays
**free** (`0 * 1.5 = 0`) and the cost grows only as the cell count moves away. An
element with no `*` carries factor `1` (neutral); each element is itself a scalar
or band, so they compose (`3,5-7*1.5`).

```bash
gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9 --target 3,5*1.5  # prefer 3, accept 5, drift past costs 1.5x
gv vad-grid  recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800 --target 3,5-7*1.5  # scaled band
```

The distance is the **min** over each element's `(raw distance × factor)`, so a
count routes through whichever element is cheapest. With `3,5*2`, the exact-
accepted count 5 stays free, but count 6 (one past) costs `1×2 = 2` while count 4
(one past the preferred 3) costs `1×1 = 1` — so the pick leans toward the
lower-factor element as cells drift away. The `best:` / `top N:` lines render the
set comma-joined with each non-neutral factor appended (`3,5*2`), the `|Δ|` shown
being the **scaled** distance, and the `--json` `target` serialises as a
`{"scaled": [[element, factor], ...]}` object (a band element nests as its own
`[lo, hi]` array). A scaled set is deduped on the element preserving first-seen
order (the first factor wins), and one that collapses to a single element drops the
now-useless factor (a lone factor scales every cell uniformly and cannot change a
pick). A `*` factor **requires** a `,` set, and cannot be combined with `>`
(preference) or `:` (the additive weight) — a set is additively *or*
multiplicatively weighted, not both, and stacking either with preference is
rejected as ambiguous. Each factor is a number `>= 1` (`1` is neutral; a factor
below 1 would *discount* an element, which the other elements' larger factors
already express); NaN and infinite factors are rejected, and an integral float
collapses to an int (`5*2.0` → `5*2`).

### `gv vad-grid` — a 2-D knob grid (iter-240)

The three `vad-sweep` axes each vary ONE knob; finding a joint elbow (e.g. "what
gate *and* hangover together?") meant running several 1-D sweeps and
cross-reading them by hand. `gv vad-grid` tabulates the cartesian product of TWO
knobs in one pass — the VAD analogue of `simulate-mirror --grid` (base_wpm ×
strength). The **row** axis is always the P(speech) gate (`--thresholds`); the
**column** axis is `--min-silences` (the hangover, the default), `--min-speeches`
(the floor), or `--speech-pads` (the symmetric region padding, iter-254) — all
millisecond knobs — or `--max-speeches` (the force-split ceiling, in *seconds*,
iter-255), mutually exclusive. The non-column knob is held at its scalar
(`--min-silence-ms` / `--min-speech-ms` / `--speech-pad-ms` / `--max-speech-s`);
every other knob is shared across all cells.

```
gv vad-grid recording.wav                                       # gate × hangover (defaults)
gv vad-grid recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800
gv vad-grid recording.wav --thresholds 0.5,0.7 --min-speeches 50,200,400  # gate × floor
gv vad-grid recording.wav --thresholds 0.5,0.7 --speech-pads 0,20,40,80   # gate × padding
gv vad-grid recording.wav --thresholds 0.5,0.7 --max-speeches 5,10,20,inf # gate × ceiling
gv vad-grid recording.wav --json                                # machine-readable cells
gv vad-grid recording.wav --csv                                 # flat CSV for plots/pivots
```

The `--speech-pads` column (iter-254) crosses the gate against the symmetric
padding Silero adds to each end of every recovered region — the 2-D counterpart
of `vad-sweep`'s fourth axis. Too little padding clips the talker's onsets and
tails; too much fuses adjacent regions until distinct utterances merge. Crossing
it against the gate exposes where that clip-vs-merge elbow shifts as the gate
tightens. The list is parsed by the same `nonneg_float_list_type` validator as
the other ms column axes and formats as bare integers (`40`, not `0.04`); the
shared `--speech-pad-ms` scalar is held fixed under the other two column axes and
ignored while `--speech-pads` is the column.

The `--max-speeches` column (iter-255) crosses the gate against the *force-split
ceiling* `max_speech_s` — Silero splits any region longer than this many seconds,
so a tight cap chops a long monologue into more (shorter) segments while `inf`
(the default) never splits. It is the only column axis measured in **seconds**,
not milliseconds, so it gets its own `max_speech_list_type` validator (each token
runs through the scalar `max_speech_type`, so the `inf`/`none`/`off` "never split"
sentinels carry through per element — include `inf` to anchor the no-cap
baseline) and its own `%g` formatter, which prints compact seconds (`5`, `12.5`)
and renders the sentinel as `inf` (no gate-style `0.00` leak). The shared
`--max-speech-s` scalar is held fixed under the three ms column axes and ignored
while `--max-speeches` is the column. Crossing the ceiling against the gate shows
where the force-split count climbs as the cap tightens, and whether a tighter gate
already keeps regions short enough that the cap never fires.

Cells are emitted in **row-major** order (each gate's full row of columns, then
the next gate). The human table is one row per cell (not a matrix) so each
cell's two metrics — segment count and speech seconds — stay unambiguous. Over
`voice-20260618-110355.wav` (gate × hangover):

```
silero VAD grid — voice-20260618-110355.wav (threshold × min_silence)
    threshold  min_silence  segments  speech
         0.30          400         5   16.1s
         0.30          800         5   17.3s
         0.50          400         5   15.7s
         0.50          800         5   16.2s
         0.70          400         6   14.2s
         0.70          800         4   15.5s
```

Each cell equals an independent `gv vad` run at that `(threshold, hangover)`
pair — the grid just segments once per cell with the shared engine, the 2-D
analogue of the `vad-sweep` row-equality property. The `--json` payload carries
both `"row_axis"` and `"col_axis"` (so a consumer knows which two dimensions the
cells vary) and a flat `"grid"` cell list keyed by those names; the `--csv`
header is `<row_axis>,<col_axis>,num_segments,speech_s` (e.g.
`threshold,min_silence_ms,…`) so the grid pivots straight into a spreadsheet.
When `max_speech_s` is the column (or row) axis the seconds cells write the bare
token `inf` for the no-cap baseline — `str(float('inf'))`, not the JSON
`Infinity` token and not a blank — once per gate row, so a multi-row grid keeps
every `inf` baseline parseable; iter-267 unit tests pin that the inf sentinel
appears in *every* threshold row and that each seconds cell `float()`-round-trips
losslessly back to its grid value across rows (the column-axis multi-row case the
iter-259 `min_silence_ms` round-trip never reached), plus a row-axis placement
proving the sentinel writes `inf` in the first column too.
Integration tests pin that every cell matches an independent `gv vad --json` run,
that recovered speech is non-increasing reading down rising thresholds *within
each column* (the gate monotonicity, now visible inside the grid), and that
`--csv` agrees with `--json` — all over the real corpus.

#### `--target` — a data-driven best-cell pick (iter-241)

The bare grid leaves the operator to eyeball which cell to pick. Passing
`--target N` (the number of speech regions you expect — e.g. one segment per
spoken sentence) surfaces a data-driven pick: the cell whose recovered segment
count is **closest** to `N`, scored by `|num_segments - N|` (lower is better).
This is the VAD counterpart of `simulate-mirror --grid`'s `pick_best_mirror_config`
best line. On an exact-distance tie the earlier cell in row-major order wins, so
a stable grid yields a stable pick.

```
gv vad-grid recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800 --target 3
```

```
silero VAD grid — voice-20260618-110355.wav (threshold × min_silence)
    threshold  min_silence  segments  speech
         0.30          400         5   16.1s
         0.30          800         5   17.3s
         0.50          400         5   15.7s
         0.50          800         5   16.2s
         0.70          400         6   14.2s
         0.70          800         4   15.5s
  best: threshold=0.70 min_silence=800 (4 segments, |Δ|=1 from target 3)
```

The trailing `best:` line names the picked `(threshold, ms)` pair, its segment
count, and the residual distance. `--json` adds a `"target"` int and a `"best"`
cell (the picked grid cell plus a `"distance"` key); `--target` is a derived
scalar, not a per-cell column, so `--csv` ignores it (the CSV stays a pure data
grid). Without `--target` the output is byte-for-byte the iter-240 shape — no
`best:` line, no `"best"`/`"target"` JSON keys. An integration test pins that the
surfaced pick genuinely minimises `|num_segments - target|` over the very grid
the run tabulated, over the real corpus.

#### `--top` — a ranked shortlist, not just the single best (iter-242)

The single `best:` pick hides the runners-up: if the winning cell sits at a knob
extreme you distrust (the highest gate, the longest hangover), you can't see how
close the next-best cell came without re-reading the table by eye. Adding
`--top K` (a positive count) lists the **K cells closest to the target**, ranked
nearest-first — the head of the shortlist is always the `best:` cell, so it
extends the pick rather than replacing it.

```
gv vad-grid recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800 --target 3 --top 3
```

```
silero VAD grid — voice-20260618-110355.wav (threshold × min_silence)
    threshold  min_silence  segments  speech
         0.30          400         5   16.1s
         0.30          800         5   17.3s
         0.50          400         5   15.7s
         0.50          800         5   16.2s
         0.70          400         6   14.2s
         0.70          800         4   15.5s
  best: threshold=0.70 min_silence=800 (4 segments, |Δ|=1 from target 3)
  top 3 (closest to target 3):
    1. threshold=0.70 min_silence=800  4 segments  |Δ|=1
    2. threshold=0.30 min_silence=400  5 segments  |Δ|=2
    3. threshold=0.30 min_silence=800  5 segments  |Δ|=2
```

The ranking sorts purely by `|num_segments - target|`, and the sort is stable,
so cells at equal distance keep their row-major order — making the shortlist head
identical to the single `best:` pick. `K` is clamped to the grid size, so a
shortlist longer than the grid simply ranks every cell. `--json` adds a `"top"`
list (each cell augmented with the same `"distance"` key, head equal to
`"best"`); like `--target`, `--top` is a derived view, not a per-cell column, so
`--csv` ignores it. `--top` rides along with `--target` — without a target there
is no distance to rank by, so the shortlist is omitted. An integration test pins
that the `K` listed distances are the `K` smallest over the whole tabulated grid,
over the real corpus.

#### `--tie-break` — a secondary ranking key (iter-243)

The `--target` pick and the `--top` shortlist rank purely on segment-count
distance; cells at equal distance fall back to **row-major order** — so the
runner-up shown first is merely the earlier cell in the grid, not necessarily the
more defensible one. When the winning cell sits at a knob extreme you distrust
(the highest gate, the longest hangover), you can't tell whether an equally-close
cell recovered more of the talker.

`--tie-break` selects how those distance ties break:

- **`row-major`** (the default) — keep the earlier grid cell, the iter-241/242
  behaviour, output unchanged byte-for-byte.
- **`speech`** — among cells equally close to the target, prefer the one that
  recovered the **most speech seconds** (it clips the talker least, so it is the
  more defensible pick than merely the earlier one).

```bash
gv vad-grid recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800 --target 3 --tie-break speech
```

Distance is always the PRIMARY key — `--tie-break speech` never lets a
farther-from-target cell win; it only re-orders cells already tied on distance.
`--json` reports the choice in a `"tie_break"` field (`"row-major"` or
`"speech"`) so a consumer knows which tie-break produced the `best`/`top`
ordering; like `--target`/`--top` it is a derived ordering, not a per-cell column,
so `--csv` ignores it, and it rides along with `--target` (no target → no pick to
order, field omitted). An integration test pins that, over the real corpus, the
`speech` pick recovers the maximum speech among all cells tied at the winning
distance.

#### `--target` on the seconds `max_speech_s` axis (iter-257)

The `--target` pick and the `--top`/`--tie-break` machinery rank purely on
`num_segments`, which is axis-agnostic, so they work unchanged when the swept (or
column) axis is the seconds force-split ceiling `--max-speeches` rather than the
gate or a millisecond knob. The only axis-specific detail is *rendering*: the
`best:` / `top N:` lines name the picked value through `_format_sweep_axis_value`,
so a seconds cap prints compactly via `%g` (`max_speech=10`, the no-cap baseline
as `max_speech=inf`) — never the gate-style `max_speech=10.00`, and never a raw
`inf.00`. Tuning the ceiling toward a desired segment count is thus a one-liner:

```bash
gv vad-sweep recording.wav --max-speeches 5,10,20,inf --target 3   # 1-D ceiling sweep
gv vad-grid  recording.wav --thresholds 0.3,0.5 --max-speeches 5,10,inf --target 3  # gate × ceiling
```

Unit tests pin that the `best:` line names the seconds value with `%g` (no
`0.00` leak) on both the 1-D sweep axis and the 2-D grid column axis, and that an
`inf` winner renders as `inf`. The `--top` shortlist rows share the same
`format_axes` closure as the `best:` line, so they inherit the same `%g`
rendering; iter-260 pins that too — each `top N:` row names its seconds cap
compactly (`max_speech=10`, the no-cap baseline as `max_speech=inf`, no `5.00` /
`inf.00` leak), on both the 1-D sweep and the 2-D grid column axis.

The same pick on the `--json` surface (iter-258) carries the bare seconds value,
not the human-formatted string: the `best` cell (and each `top` cell) holds
`"max_speech_s": 10.0`, with the no-cap baseline as the JSON `Infinity` token,
which round-trips back to `float('inf')` through `json.loads`. So a consumer
reading `payload["best"]["max_speech_s"]` gets a number it can compare and plot
directly — no `%g` parsing step — and the `inf` sentinel survives unchanged:

```bash
gv vad-sweep recording.wav --max-speeches 5,10,inf --target 2 --json   # best.max_speech_s == 10.0
gv vad-grid  recording.wav --thresholds 0.3 --max-speeches 5,10,inf --target 2 --json
```

Unit tests pin the JSON `best`/`top` cells on both surfaces, including that the
`inf` baseline survives the round-trip.

The third machine surface, `--csv` (iter-259), writes the raw seconds cap into
the first column via the stdlib `csv` writer: a finite cap renders as `10.0` and
the no-cap baseline as the bare token `inf` (`str(float('inf'))`) — *not* the
JSON-style `Infinity`, and never a blank cell. So a `loadtxt`/`read_csv` consumer
recovers the seconds axis losslessly (every cap cell parses back through
`float(...)`, `inf` included), and the CSV stays self-describing across the
no-cap baseline:

```bash
gv vad-sweep recording.wav --max-speeches 5,10,inf --csv               # first column: 5.0 / 10.0 / inf
gv vad-grid  recording.wav --thresholds 0.3 --max-speeches 5,10,inf --csv
```

Unit tests pin the `inf` baseline writing as `inf` (not `Infinity`, not blank) on
both the 1-D sweep axis and the 2-D grid column axis, and that every cap cell
round-trips through `float(...)`.

The `--tie-break speech` secondary key (iter-243) is orthogonal to the axis: it
breaks distance ties on recovered speech (most first) regardless of which knob is
swept. iter-261 pins the two seams *together* on the seconds axis — when two caps
tie on segment count, `--tie-break speech` names the cap that recovers the most
speech (e.g. the no-cap `inf` baseline over a clipping finite cap), and that
winner still renders compactly (`max_speech=inf`, never `inf.00`; the finite cap
never `10.00`) on the `best:` line *and* the `--top` shortlist rows, across both
the 1-D sweep and the 2-D grid column axis. The default `row-major` tie-break
keeps the earlier finite cap, proving the seconds axis honours the
earliest-tie rule like every other axis:

```bash
gv vad-sweep recording.wav --max-speeches 10,inf --target 3 --tie-break speech  # most-speech cap wins
gv vad-grid  recording.wav --thresholds 0.3 --max-speeches 5,inf --target 3 --top 2 --tie-break speech
```

The banded `--target lo-hi` form (iter-246) — a count *window* scoring distance
0 for any cell inside the band, else distance to the nearer edge — and its
open-edge variants (`lo-` "at least", `-hi` "at most", iter-247) are likewise
orthogonal to the swept knob. iter-262 pins the band-scoring path *together* with
the seconds axis: a band on `max_speech_s` picks the in-band cap, renders the
band as `3-5` / `3-` / `-1` (never a `(3, 5)` tuple repr or a `None` leak), and
names the chosen SECONDS cap compactly (`max_speech=10`, the no-cap baseline as
`max_speech=inf`, never `10.00` / `inf.00`) on the `best:` line across both the
1-D sweep and the 2-D grid column axis. The grid JSON surface carries the band as
a `[lo, hi]` array and emits the chosen cap as a finite seconds number
(`best.max_speech_s == 5.0`):

```bash
gv vad-sweep recording.wav --max-speeches 5,10,inf --target 3-5            # in-band cap wins
gv vad-grid  recording.wav --thresholds 0.3 --max-speeches 5,inf --target 3-5 --json
```

The comma SET form (`--target 2,4,6`, iter-248) — where a cell scores its
distance to the NEAREST listed element, so any cell landing on a listed count
scores 0 — is the same shape of orthogonal seam. iter-263 pins it *together*
with the seconds axis: a set on `max_speech_s` picks the cap that lands on (or
nearest) a listed count, renders the set as `2,4,6` (never a `[2, 4, 6]` list
repr), and names the chosen SECONDS cap compactly (`max_speech=10`, the no-cap
baseline as `max_speech=inf`, never `10.00` / `inf.00`) on the `best:` line
across both the 1-D sweep and the 2-D grid column axis. The grid JSON surface
carries the set as a JSON array and emits the chosen cap as a finite seconds
number (`best.max_speech_s == 5.0`):

```bash
gv vad-sweep recording.wav --max-speeches 5,10,inf --target 2,4,6          # cap on a listed count wins
gv vad-grid  recording.wav --thresholds 0.3 --max-speeches 5,inf --target 2,4,6 --json
```

The ranked PREFERENCE form (`--target 4>2`, iter-249) is the same shape of seam
with one twist: its DISTANCE is the MIN over its elements (identical to the flat
set), but its precedence breaks EXACT distance ties toward the earlier-listed
(more-preferred) element. iter-264 pins it *together* with the seconds axis: when
two caps both land on a preference element (both `|Δ|=0`), the preference picks
the more-preferred count's cap even when it is NOT the earliest row — so
`--target 4>2` over caps that recover 2 and 4 segments picks the 4-segment cap,
where the flat set `4,2` would pick the earlier row instead. The `best:` line
renders the preference as `4>2` (never a `{"prefer": ...}` dict repr) and names
the chosen SECONDS cap compactly (`max_speech=10`, the no-cap baseline as
`max_speech=inf`, never `10.00` / `inf.00`) across both the 1-D sweep and the 2-D
grid column axis. The grid JSON surface carries the preference as its
`{"prefer": [...]}` dict (distinct from a flat-set array) and emits the chosen cap
as a finite seconds number (`best.max_speech_s == 5.0`):

```bash
gv vad-sweep recording.wav --max-speeches 5,10,inf --target 4>2            # preferred count's cap wins the tie
gv vad-grid  recording.wav --thresholds 0.3 --max-speeches 5,inf --target 4>2 --json
```

The additive-penalty WEIGHTED set (`--target 3,6:2`, iter-250) is the stronger
cousin of the preference: where the preference folds intent only into the
tie-break, the weighted set folds a per-element penalty into the DISTANCE itself
(the score is the MIN over each element's raw distance PLUS its penalty), so it
can override a distance GAP, not merely an exact tie. iter-265 pins it *together*
with the seconds axis: `--target 3,6:2` makes the bare `3` free and the `6` cost
`+2`, so a cap one segment off the free `3` (penalised `1`) beats a cap sitting
exactly on the costly `6` (penalised `2`) — the flat set `3,6` would pick the
on-`6` cap instead (raw distance `0`). The `best:` line renders the weighted set
as `3,6:2` (never a `{"weighted": ...}` dict repr) and names the chosen SECONDS
cap compactly (`max_speech=10` / `max_speech=5`, the no-cap baseline as
`max_speech=inf`, never `10.00` / `inf.00`) across both the 1-D sweep and the 2-D
grid column axis. The grid JSON surface carries the weighted set as its
`{"weighted": [[element, penalty], ...]}` dict (each pair a 2-element array,
distinct from a flat-set array of scalars) and emits the chosen cap as a finite
seconds number (`best.max_speech_s == 5.0`) with the penalised distance:

```bash
gv vad-sweep recording.wav --max-speeches 5,10,inf --target 3,6:2          # +2 penalty overrides the raw-distance gap
gv vad-grid  recording.wav --thresholds 0.3 --max-speeches 5,inf --target 3,6:2 --json
```

The multiplicative-factor SCALED set (`--target 3,8*2`, iter-252) is the
multiplicative twin of the weighted set: where the weighted set ADDS a constant
per-element penalty, the scaled set MULTIPLIES each element's raw distance by its
factor (the score is the MIN over each element's raw distance TIMES its factor).
Two consequences distinguish it from BOTH the flat set and the additive weighted
set: an exact hit stays FREE (raw `0` × any factor = `0`, so a high factor never
bites an on-target cap — unlike an additive penalty), and the cost GROWS with
distance (one count off a factor-`2` element costs `2`, two counts cost `4`, not
the constant offset the weighted form adds). iter-266 pins it *together* with the
seconds axis: `--target 3,8*2` makes the bare `3` free and AMPLIFIES drift past
the `8`, so a cap one segment off the free `3` beats a cap four off the `8` even
though the flat set `3,8` would tie them — while a cap landing exactly on the
costly `8` still scores `0` (the additive `3,8:2` would instead penalise it to
`2`). The `best:` line renders the scaled set as `3,8*2` (never a
`{"scaled": ...}` dict repr) and names the chosen SECONDS cap compactly
(`max_speech=10` / `max_speech=5`, the no-cap baseline as `max_speech=inf`, never
`10.00` / `inf.00`) across both the 1-D sweep and the 2-D grid column axis. The
grid JSON surface carries the scaled set as its `{"scaled": [[element, factor],
...]}` dict (each pair a 2-element array, distinct from a flat-set array of
scalars and from a `{"weighted": ...}` dict) and emits the chosen cap as a finite
seconds number (`best.max_speech_s == 5.0`) with the scaled distance:

```bash
gv vad-sweep recording.wav --max-speeches 10,inf --target 3,8*2            # factor amplifies the off-8 gap, picks the finite cap
gv vad-grid  recording.wav --thresholds 0.3 --max-speeches 5,inf --target 3,8*2 --json
```

### Silero vs energy-VAD segment counts (the headless proof)

Measured over the seed corpus with `min_silence_ms=800` (the pipecat
`stop_secs=0.8`), Silero defaults otherwise (`threshold=0.5`,
`min_speech_ms=250`, `speech_pad_ms=30`):

| recording | dur | energy-VAD onsets | **Silero segments** |
|-----------|----:|------------------:|--------------------:|
| voice-20260617-122716.wav | 17.3s | 2 | 2 |
| voice-20260617-123829.wav | 64.6s | 2 | **4** |
| voice-20260617-131451.wav |  9.3s | 1 | 1 |
| voice-20260617-135015.wav | 1115.8s | 5 | 3 |
| voice-20260617-161615.wav | 12.4s | 1 | **2** |
| **voice-20260618-110355.wav** | **31.3s** | **1** | **5** |

The gate the steering set: the 31s continuous recording, which energy-VAD
collapses to a single never-closing segment, splits into **5** sensible Silero
regions (e.g. `(1.6-2.1) (3.9-4.4) (6.8-7.7) (10.7-18.5) (24.8-31.3)`s). Two
more recordings (`123829`, `161615`) that energy-VAD under-segments also gain
correct turn boundaries. `135015` (an 18-minute mostly-silent capture) shows
the opposite, expected behaviour: Silero reports *fewer* regions because it
rejects the low-energy noise the RMS gate spuriously committed on. The
`test_silero_recordings.py` integration suite pins all of this; the pure
plumbing is covered fast in `tests/unit/test_silero_vad.py`.

### Endpoint contract (`POST /vad/silero`, server :5111)

So the desktop client can adopt it:

- **Request:** `POST /vad/silero` with a 16-bit PCM WAV byte body (any sample
  rate / channel count; resampled to 16kHz mono internally). Optional query
  params override the `SileroParams`: `threshold` (speech-probability gate,
  default 0.5), `min_speech_ms` (250), `min_silence_ms` (800), `speech_pad_ms`
  (30).
- **Response (200):** `{"num_segments": N, "speech_s": S, "duration_s": D,
  "sample_rate": SR, "segments": [{"start_s", "end_s", "duration_s"}, ...]}` —
  timestamps in seconds relative to the recording start.
- **503** when the Silero model is unavailable (package not installed) — the
  caller falls back to the local RMS path.
- `/health` now reports `"silero_vad": true|false` so a client can probe
  availability before choosing the endpoint over local RMS.

### Streaming endpoint contract (`WS /vad/silero/stream`, iter-232)

The `POST /vad/silero` above is **batch**: it needs the whole utterance buffered
before it returns a single segment list. For live capture the desktop client
wants *incremental* decisions — a turn cut the instant Silero sees
`min_silence_ms` of trailing silence, not after the user stops AND a whole-WAV
round-trip. `WS /vad/silero/stream` exposes the `silero-vad` `VADIterator`
"stream imitation" for exactly this:

- **Config (optional first text frame):** JSON overriding `SileroParams` —
  `{"threshold":0.5,"min_silence_ms":800,"speech_pad_ms":30,"sample_rate":16000}`.
  Send before any audio; a binary frame sent first arms a default (16 kHz)
  stream. **`min_speech_ms` / `max_speech_s` do NOT apply** on the streaming path
  — `VADIterator` has no look-back to retroactively drop a region that turns out
  too short, so the stream is a **superset** of the batch path (it may emit
  sub-`min_speech_ms` blips the batch `get_speech_timestamps` would filter). The
  integration test pins this: stream output filtered to `>= min_speech_ms`
  reconstructs the batch segmentation exactly.
- **Audio (binary frames):** little-endian float32 mono PCM at `sample_rate` (no
  server-side resampling — feed it the model's 16 kHz). Any length; sub-window
  remainders are buffered across frames, so a mic callback's frame size is fine.
- **Per-frame reply:** `{"events":[{"type":"start"|"end","time_s":...}, ...]}`
  (possibly empty) — `start` when speech opens, `end` when it closes after the
  silence hangover. Timestamps match the batch path's seconds.
- **`{"cmd":"flush"}`:** closes a region still open because the audio ended
  mid-speech → `{"events":[...],"flushed":true}`. (The batch path gets this for
  free; a live stream must flush explicitly — see `SileroStream.flush`.)
- **`{"cmd":"reset"}`:** re-arms the same stream for a new utterance without
  reloading the model → `{"events":[],"reset":true}`.
- **1011 close** when the Silero model is unavailable.

The window quantum is **512 samples @ 16 kHz** (32 ms) / 256 @ 8 kHz — the only
sizes `VADIterator` accepts; `SileroStream` buffers to it internally. The
message-protocol state machine is `vad.StreamProtocol` (unit-tested without a
socket); the server endpoint is thin transport glue with `push` off-loaded to
the executor.

### Desktop integration (backlog — operator wires + GUI-tests on the Mac)

The browser `ContinuousListener` should send mic audio to `/vad/silero` (batch)
or **`/vad/silero/stream`** (live, iter-232) for speech/silence decisions
instead of (or alongside) local RMS — OR the app uses the existing pipecat
`:8765` WS path, which already runs Silero. For live capture prefer
`/vad/silero/stream` (or the pipecat WS): both run the model frame-by-frame so a
turn cuts without buffering the whole utterance. The batch `POST /vad/silero`
stays the simplest contract for whole-clip segmentation. Energy VAD (threshold /
gain / preroll, below) stays as the documented **fallback** path for hosts
without `silero-vad`.

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
