"""Adapter contract tests that never load real speech models."""

from __future__ import annotations

import asyncio
import io
import sys
import types
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from geno_voice.endpoint.models.breeze import BreezeTTS2Adapter
from geno_voice.endpoint.models.kokoro import KokoroAdapter
from geno_voice.endpoint.types import CancellationToken, SynthesisRequest


@dataclass
class FakeBreezeChunk:
    audio: np.ndarray
    sample_rate: int = 24_000
    is_final: bool = True


@pytest.fixture
def fake_breeze_modules(monkeypatch):
    calls: dict[str, object] = {}
    tokenizer = object()
    model = object()
    audio_tokenizer = object()

    runtime_module = types.ModuleType("breeze_infer.runtime")

    def load_runtime(path, *, device, attn_implementation):
        calls["load_runtime"] = (path, device, attn_implementation)
        return tokenizer, model, audio_tokenizer

    runtime_module.load_runtime = load_runtime
    runtime_module.resolve_device = lambda: "cuda:0"
    runtime_module.update_generation_config_for_breeze = (
        lambda loaded_model: calls.setdefault("updated_model", loaded_model)
    )

    templates_module = types.ModuleType("breeze_infer.templates")
    templates_module.get_template = lambda name: f"template:{name}"

    def prepare_inputs(*args, **kwargs):
        calls["prepare_inputs"] = (args, kwargs)
        return {"prepared": True}

    templates_module.prepare_inputs = prepare_inputs

    fast_module = types.ModuleType("models.fast_streaming")

    class FastStreamingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FastBreezeStreamingRuntime:
        def __init__(self, loaded_model, loaded_audio_tokenizer, config, *, tokenizer):
            calls["runtime_init"] = (
                loaded_model,
                loaded_audio_tokenizer,
                config,
                tokenizer,
            )

        def iter_audio_chunks(self, inputs, *, request_id):
            calls["iter_audio_chunks"] = (inputs, request_id)
            yield FakeBreezeChunk(np.array([1.0, -1.0], dtype=np.float32))

    fast_module.FastStreamingConfig = FastStreamingConfig
    fast_module.FastBreezeStreamingRuntime = FastBreezeStreamingRuntime

    packages = {
        "breeze_infer": types.ModuleType("breeze_infer"),
        "breeze_infer.runtime": runtime_module,
        "breeze_infer.templates": templates_module,
        "models": types.ModuleType("models"),
        "models.fast_streaming": fast_module,
    }
    packages["breeze_infer"].__path__ = []
    packages["models"].__path__ = []
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_breeze_adapter_uses_official_runtime_and_converts_float_pcm(
    fake_breeze_modules, tmp_path
) -> None:
    async def scenario() -> None:
        runtime_path = tmp_path / "breeze-runtime"
        runtime_path.mkdir()
        adapter = BreezeTTS2Adapter(
            model_path=Path("/models/breeze"), runtime_path=runtime_path
        )

        await adapter.load()
        chunks = [
            chunk
            async for chunk in adapter.synthesize(
                SynthesisRequest(
                    request_id="r1",
                    text="Hello",
                    instruction="Warm and calm",
                ),
                CancellationToken(),
            )
        ]

        assert chunks[0].pcm == b"\xff\x7f\x01\x80"
        assert chunks[0].final is True
        assert fake_breeze_modules["load_runtime"] == (
            Path("/models/breeze"),
            "cuda:0",
            "eager",
        )
        prepare_args, prepare_kwargs = fake_breeze_modules["prepare_inputs"]
        assert prepare_args[3] == [
            {
                "id": "r1",
                "text": "Hello",
                "instruction": "Warm and calm",
                "speaker": "S0",
            }
        ]
        assert prepare_args[4] == "template:tts_instruction"
        assert prepare_kwargs == {
            "guidance_scale": 1.0,
            "guidance_scale_ref": None,
            "guidance_scale_ins": None,
        }
        assert str(runtime_path) not in sys.path

    asyncio.run(scenario())


def test_breeze_adapter_requires_a_model_path() -> None:
    async def scenario() -> None:
        adapter = BreezeTTS2Adapter(model_path=None)
        with pytest.raises(ValueError, match="--model-path"):
            await adapter.load()

    asyncio.run(scenario())


def make_wav(pcm: bytes, *, sample_rate: int = 24_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


class FakeKokoroEngine:
    def __init__(self, wav_chunks: list[bytes]) -> None:
        self.wav_chunks = wav_chunks
        self.calls: list[tuple[str, str, float]] = []

    def stream(self, text: str, voice: str, speed: float):
        self.calls.append((text, voice, speed))
        yield from self.wav_chunks


def test_kokoro_adapter_unwraps_wav_chunks_without_blocking_contract() -> None:
    async def scenario() -> None:
        engine = FakeKokoroEngine([make_wav(b"\x01\x00\x02\x00")])
        adapter = KokoroAdapter(engine=engine, default_voice="af_heart")

        chunks = [
            chunk
            async for chunk in adapter.synthesize(
                SynthesisRequest(request_id="r1", text="Hi", speed=1.25),
                CancellationToken(),
            )
        ]

        assert [chunk.pcm for chunk in chunks] == [b"\x01\x00\x02\x00"]
        assert engine.calls == [("Hi", "af_heart", 1.25)]

    asyncio.run(scenario())


def test_kokoro_adapter_rejects_noncanonical_wav() -> None:
    async def scenario() -> None:
        engine = FakeKokoroEngine([make_wav(b"\x01\x00", sample_rate=16_000)])
        adapter = KokoroAdapter(engine=engine)

        with pytest.raises(ValueError, match="24000 Hz"):
            async for _ in adapter.synthesize(
                SynthesisRequest(request_id="r1", text="Hi"),
                CancellationToken(),
            ):
                pass

    asyncio.run(scenario())
