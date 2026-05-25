"""Tests for iter-081 — filler novelty index.

Metric 3.8 from docs/perf-metrics-taxonomy.md.

    novelty = unique({m.last_filler_id for m if id != 0}) / fillers_total

Hearing the same "umm" three times in a row is worse than no
filler at all. The metric measures whether the filler picker is
actually distributing across the rendered_fillers list.
"""

from __future__ import annotations

import io
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)
from examples._chat_pipeline import SentenceWorker  # noqa: E402


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


def _m(played=0, fid=0):
    return TurnMetrics(ttfs=0.5, fillers_played=played, last_filler_id=fid)


# ---- Default + worker latching --------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().last_filler_id == 0

    def test_worker_default_none(self):
        w = SentenceWorker(
            speaker_factory=lambda: SimpleNamespace(
                write=lambda b: None, close=lambda: None,
            ),
            synth_fn=lambda s: (np.zeros(8, dtype=np.float32), []),
            play_fn=lambda *a, **k: 0.0,
        )
        assert w.last_filler_id is None


class TestWorkerLatch:
    def _make_worker(self, *, fillers, picker):
        # The filler fires when queue.get(timeout=idle_threshold)
        # raises Empty. Set idle_threshold low so the timeout fires
        # quickly. Keep ``submit_done()`` delayed so the timeout
        # path actually triggers — sending the sentinel immediately
        # would beat the timeout.
        return SentenceWorker(
            speaker_factory=lambda: SimpleNamespace(
                write=lambda b: None, close=lambda: None,
            ),
            synth_fn=lambda s: (np.zeros(8, dtype=np.float32), []),
            play_fn=lambda *a, **k: 0.0,
            fillers=fillers,
            idle_threshold=0.02,
            filler_picker=picker,
        )

    def _run_with_filler(self, w):
        # Start, give the filler timeout window time to fire
        # (50ms — well past the 20ms threshold), THEN signal done.
        w.start()
        def _later():
            time.sleep(0.05)
            w.submit_done()
        threading.Thread(target=_later, daemon=True).start()
        w.wait_done(timeout=2.0)

    def test_no_fillers_keeps_none(self):
        # Empty fillers list — no filler can play, last_filler_id
        # stays None.
        w = self._make_worker(fillers=[], picker=lambda lst: None)
        self._run_with_filler(w)
        assert w.last_filler_id is None

    def test_single_filler_play_latches_id(self):
        # One filler available; it's the only thing the picker can
        # return. Worker should latch id(clip) on play.
        clip = (np.zeros(8, dtype=np.float32), [])
        w = self._make_worker(
            fillers=[clip],
            picker=lambda lst: lst[0],  # always return the one clip
        )
        self._run_with_filler(w)
        assert w.last_filler_id == id(clip)

    def test_filler_picker_picks_one(self):
        # 3 fillers available. picker returns the LAST item in
        # whatever list it sees so iter-087's multi-shot path
        # (which filters out already-played clips) is observable:
        # first fire picks clips[2]; second fire's filtered list is
        # [clips[0], clips[1]] → picks clips[1]. ``last_filler_id``
        # ends up at the last played, which is clips[1].
        clips = [
            (np.full(8, 0.1, dtype=np.float32), []),
            (np.full(8, 0.2, dtype=np.float32), []),
            (np.full(8, 0.3, dtype=np.float32), []),
        ]
        w = self._make_worker(fillers=clips, picker=lambda lst: lst[-1])
        self._run_with_filler(w)
        # last_filler_id must be one of the picked clips. Allow
        # either single-shot (clips[2]) or multi-shot (clips[1])
        # outcomes — both are valid depending on how the timing
        # raced inside the 50ms wait window.
        assert w.last_filler_id in {id(clips[1]), id(clips[2])}


# ---- Session aggregate ----------------------------------------


class TestSessionSummary:
    def test_no_fillers_omits(self):
        plain = _summary([_m(), _m()])
        assert "Filler novelty" not in plain

    def test_single_filler_omits(self):
        # Single play — novelty trivially 100%, skip the line.
        plain = _summary([_m(played=1, fid=12345)])
        assert "Filler novelty" not in plain

    def test_perfect_diversity(self):
        # 3 plays, 3 distinct ids → 100%.
        plain = _summary([
            _m(played=1, fid=111),
            _m(played=1, fid=222),
            _m(played=1, fid=333),
        ])
        assert "Filler novelty:   3 unique / 3 (100%)" in plain

    def test_partial_diversity(self):
        # 4 plays, 2 distinct ids → 50%.
        plain = _summary([
            _m(played=1, fid=111),
            _m(played=1, fid=222),
            _m(played=1, fid=111),  # repeat
            _m(played=1, fid=222),  # repeat
        ])
        assert "Filler novelty:   2 unique / 4 (50%)" in plain

    def test_all_same_clip_low_novelty(self):
        # 5 plays of the same clip → 1/5 = 20%.
        plain = _summary([_m(played=1, fid=999) for _ in range(5)])
        assert "Filler novelty:   1 unique / 5 (20%)" in plain

    def test_zero_filler_id_excluded(self):
        # Turns with last_filler_id==0 are "no filler played" —
        # excluded from the unique-count denominator (they had
        # nothing to deduplicate).
        plain = _summary([
            _m(played=1, fid=111),
            _m(played=1, fid=222),
            _m(played=0, fid=0),    # no filler this turn
        ])
        # 2 unique / 2 total filler plays = 100%.
        assert "Filler novelty:   2 unique / 2 (100%)" in plain
