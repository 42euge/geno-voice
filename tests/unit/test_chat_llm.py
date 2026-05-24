"""Tests for examples/_chat_llm.py — SSE parser + close cascade.

Two layers:

  1. ``parse_sse_token_stream`` — pure parsing logic. Drive it with
     synthetic line iterables and assert the yielded tokens match.
  2. The close-cascade contract — when the outer ``stream_chat_completion``
     generator is closed (via ``BargeInCoordinator.on_trigger`` in
     production), GeneratorExit propagates through to the
     ``finally`` block which closes the underlying HTTP response.
     Tested with a fake response object that records the close.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_llm import (  # noqa: E402
    parse_sse_token_stream,
    stream_chat_completion,
)
from examples._chat_pipeline import BargeInCoordinator  # noqa: E402


# ---- parse_sse_token_stream --------------------------------------------------


def _data_line(content: str) -> str:
    payload = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(payload)}"


class TestParseSse:
    def test_empty_input_yields_nothing(self):
        assert list(parse_sse_token_stream([])) == []

    def test_single_token_line(self):
        out = list(parse_sse_token_stream([_data_line("hello")]))
        assert out == ["hello"]

    def test_multiple_tokens_in_order(self):
        lines = [_data_line(t) for t in ["The", " quick", " brown", " fox"]]
        assert list(parse_sse_token_stream(lines)) == [
            "The", " quick", " brown", " fox",
        ]

    def test_done_sentinel_stops_iteration(self):
        lines = [
            _data_line("alpha"),
            "data: [DONE]",
            _data_line("never seen"),
        ]
        assert list(parse_sse_token_stream(lines)) == ["alpha"]

    def test_blank_lines_skipped(self):
        lines = [
            "",
            _data_line("hi"),
            "",
            _data_line("there"),
            "",
        ]
        assert list(parse_sse_token_stream(lines)) == ["hi", "there"]

    def test_non_data_lines_skipped(self):
        # SSE keep-alive comments start with ":". Event-name lines
        # begin with "event:". Both should be silently ignored.
        lines = [
            ":heartbeat",
            "event: chunk",
            _data_line("real"),
        ]
        assert list(parse_sse_token_stream(lines)) == ["real"]

    def test_malformed_json_data_skipped(self):
        lines = [
            "data: {not json",
            _data_line("ok"),
            "data: [also bogus",
        ]
        assert list(parse_sse_token_stream(lines)) == ["ok"]

    def test_missing_choices_field_skipped(self):
        lines = [
            "data: " + json.dumps({"foo": "bar"}),  # no choices
            _data_line("recovers"),
        ]
        assert list(parse_sse_token_stream(lines)) == ["recovers"]

    def test_missing_delta_content_skipped(self):
        # delta exists but has no content → skipped (some providers
        # emit role/finish_reason chunks with no content).
        role_chunk = json.dumps(
            {"choices": [{"delta": {"role": "assistant"}}]}
        )
        lines = [f"data: {role_chunk}", _data_line("payload")]
        assert list(parse_sse_token_stream(lines)) == ["payload"]

    def test_empty_string_content_skipped(self):
        lines = [
            _data_line(""),
            _data_line("nonempty"),
        ]
        assert list(parse_sse_token_stream(lines)) == ["nonempty"]

    def test_bytes_lines_decoded(self):
        line = _data_line("byteline").encode("utf-8")
        assert list(parse_sse_token_stream([line])) == ["byteline"]

    def test_invalid_utf8_in_bytes_does_not_crash(self):
        # 0xff is not valid utf-8; errors='replace' should preserve
        # whatever's still recognizable. This case has a malformed
        # data line (it can't be valid JSON after replacement), so
        # we expect 0 tokens, not a raised exception.
        bad = b"data: " + b"\xff\xfe garbage"
        # Should produce no tokens and not raise.
        assert list(parse_sse_token_stream([bad])) == []

    def test_index_error_on_empty_choices_skipped(self):
        # ``choices: []`` — chunk["choices"][0] raises IndexError;
        # the parser should swallow it.
        empty_choices = json.dumps({"choices": []})
        lines = [f"data: {empty_choices}", _data_line("after")]
        assert list(parse_sse_token_stream(lines)) == ["after"]


# ---- close cascade -----------------------------------------------------------


class FakeResponse:
    """A pyaudio-stream-style fake of ``requests.Response`` — exposes
    ``iter_lines`` and ``close``. ``raise_for_status`` is a no-op so
    we don't raise on status codes.
    """

    def __init__(self, lines: list, *, slow_per_line: float = 0.0):
        self._lines = list(lines)
        self.closed = False
        self.close_calls = 0
        self._slow_per_line = slow_per_line

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for line in self._lines:
            if self.closed:
                return
            if self._slow_per_line:
                time.sleep(self._slow_per_line)
            yield line

    def close(self):
        self.close_calls += 1
        self.closed = True


def _patch_post(monkeypatch, response: FakeResponse):
    """Patch ``requests.post`` (as imported inside stream_chat_completion)
    to return our fake.
    """
    import requests

    def fake_post(*args, **kwargs):
        return response

    monkeypatch.setattr(requests, "post", fake_post)
    return response


@pytest.fixture
def llm_config():
    return {
        "api_key": "fake",
        "base_url": "http://localhost",
        "model": "fake-model",
        "max_tokens": 128,
    }


class TestStreamChatCompletionLifecycle:
    def test_full_consumption_closes_response(self, monkeypatch, llm_config):
        resp = FakeResponse([
            _data_line("hello"),
            _data_line(" world"),
            "data: [DONE]",
        ])
        _patch_post(monkeypatch, resp)

        out = list(stream_chat_completion([], llm_config))
        assert out == ["hello", " world"]
        # finally block ran, response closed.
        assert resp.close_calls == 1
        assert resp.closed is True

    def test_generator_close_propagates_to_response_close(
        self, monkeypatch, llm_config
    ):
        # Lots of lines so we can pause mid-stream and call close.
        many_tokens = [_data_line(f"t{i}") for i in range(100)]
        resp = FakeResponse(many_tokens, slow_per_line=0.005)
        _patch_post(monkeypatch, resp)

        gen = stream_chat_completion([], llm_config)
        # Pull a few tokens to start the generator inside iter_lines.
        first = next(gen)
        assert first == "t0"
        # Now close the generator from outside — finally must run.
        gen.close()
        assert resp.closed is True
        assert resp.close_calls == 1

    def test_response_close_exception_does_not_propagate(
        self, monkeypatch, llm_config
    ):
        class BoomResponse(FakeResponse):
            def close(self):
                self.close_calls += 1
                raise RuntimeError("close boom")

        resp = BoomResponse([_data_line("a"), "data: [DONE]"])
        _patch_post(monkeypatch, resp)

        # Consuming the full stream should not raise even though
        # close() raises in the finally.
        out = list(stream_chat_completion([], llm_config))
        assert out == ["a"]
        assert resp.close_calls == 1


class TestConsumerSideCloseReleasesResponse:
    """The mic_chat pattern: the consumer of the generator wraps its
    for-loop in ``try/finally: gen.close()``. When the consumer
    breaks (whether due to barge-in or normal completion), the
    finally block runs in the same thread and the generator's own
    finally block closes the response.

    Cross-thread ``gen.close()`` from a watcher thread doesn't
    work — Python raises ``ValueError("generator already
    executing")`` if the consumer is mid-``next()``, and the
    coordinator's hook-exception swallow turns that into a silent
    no-op. So we test the same-thread pattern that mic_chat uses.
    """

    def test_consumer_break_with_finally_close_releases_response(
        self, monkeypatch, llm_config
    ):
        many = [_data_line(f"t{i}") for i in range(20)]
        resp = FakeResponse(many)
        _patch_post(monkeypatch, resp)

        gen = stream_chat_completion([], llm_config)
        coord = BargeInCoordinator(worker=None)

        consumed: list[str] = []
        try:
            for tok in gen:
                consumed.append(tok)
                if len(consumed) == 3:
                    coord.trigger()  # simulate barge-in
                    break
        finally:
            gen.close()

        assert coord.is_set()
        assert consumed == ["t0", "t1", "t2"]
        # finally + gen.close both invoked the response close
        # (idempotent on the response side: closed stays True).
        assert resp.closed is True
        assert resp.close_calls >= 1

    def test_normal_completion_closes_response_via_finally(
        self, monkeypatch, llm_config
    ):
        """No barge-in: the generator runs to [DONE], its own
        try/finally closes the response. Idempotent — calling
        gen.close() afterward is a no-op on the resp side
        because closed is already True.
        """
        resp = FakeResponse([_data_line("x"), "data: [DONE]"])
        _patch_post(monkeypatch, resp)

        gen = stream_chat_completion([], llm_config)
        out = list(gen)
        assert out == ["x"]
        # Generator's finally block already closed the response.
        assert resp.closed is True
        assert resp.close_calls == 1

        # Defensive close (mic_chat does this) — no extra side effects.
        gen.close()
        assert resp.closed is True
        # Some Python versions invoke close again; bound loosely.
        assert resp.close_calls <= 2
