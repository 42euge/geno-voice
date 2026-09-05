"""Breeze-TTS-2 adapter for the official CUDA streaming runtime."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..types import (
    AudioChunk,
    CancellationToken,
    ModelCapabilities,
    SynthesisRequest,
)
from . import stream_sync_iterator


@contextmanager
def _temporary_import_path(path: Path | None):
    if path is None:
        yield
        return
    value = str(path.resolve())
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


class BreezeTTS2Adapter:
    name = "breeze-tts-2"
    capabilities = ModelCapabilities(
        streaming=True,
        voice_cloning=True,
        voice_design=True,
    )
    license_warning = (
        "Breeze-TTS-2 weights and self-hosted outputs are research/non-commercial "
        "under the BreezeBlue Research and Non-Commercial License."
    )

    def __init__(
        self,
        *,
        model_path: Path | str | None,
        runtime_path: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.runtime_path = Path(runtime_path) if runtime_path is not None else None
        self.device = device
        self._tokenizer: Any = None
        self._model: Any = None
        self._audio_tokenizer: Any = None
        self._runtime: Any = None
        self._prepare_inputs: Any = None
        self._get_template: Any = None
        self._set_all_seeds: Any = None
        self._inference_lock = asyncio.Lock()

    async def load(self) -> None:
        if self.model_path is None:
            raise ValueError("Breeze-TTS-2 requires --model-path")
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        with _temporary_import_path(self.runtime_path):
            try:
                from breeze_infer.runtime import (
                    load_runtime,
                    resolve_device,
                    set_all_seeds,
                    update_generation_config_for_breeze,
                )
                from breeze_infer.templates import get_template, prepare_inputs
                from models.fast_streaming import (
                    FastBreezeStreamingRuntime,
                    FastStreamingConfig,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Breeze official runtime is not importable; install "
                    "https://github.com/breezeblue-ai/breeze-tts or pass --runtime-path"
                ) from exc

            device = self.device or resolve_device()
            tokenizer, model, audio_tokenizer = load_runtime(
                self.model_path,
                device=device,
                attn_implementation="eager",
            )
            update_generation_config_for_breeze(model)
            config = FastStreamingConfig(
                max_new_tokens=1_500,
                max_seq_len=2_048,
                repetition_penalty=1.1,
            )
            runtime = FastBreezeStreamingRuntime(
                model, audio_tokenizer, config, tokenizer=tokenizer
            )

        self.device = device
        self._tokenizer = tokenizer
        self._model = model
        self._audio_tokenizer = audio_tokenizer
        self._runtime = runtime
        self._prepare_inputs = prepare_inputs
        self._get_template = get_template
        self._set_all_seeds = set_all_seeds

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ):
        if self._runtime is None:
            raise RuntimeError("Breeze-TTS-2 adapter has not been loaded")

        reference_path: Path | None = None
        if request.reference_audio is not None:
            with tempfile.NamedTemporaryFile(
                prefix="geno_breeze_ref_", suffix=".wav", delete=False
            ) as reference_file:
                reference_file.write(request.reference_audio)
                reference_path = Path(reference_file.name)

        try:
            async with self._inference_lock:
                inputs = await asyncio.to_thread(
                    self._build_inputs, request, reference_path
                )

                def chunks():
                    self._set_all_seeds(42)
                    return self._runtime.iter_audio_chunks(
                        inputs, request_id=request.request_id, seed=42
                    )

                async for chunk in stream_sync_iterator(chunks, cancellation):
                    if chunk.sample_rate != 24_000:
                        raise ValueError(
                            f"Breeze emitted {chunk.sample_rate} Hz audio; expected 24000 Hz"
                        )
                    pcm = self._float_to_pcm16(chunk.audio)
                    if pcm:
                        yield AudioChunk(
                            pcm=pcm,
                            sample_rate=chunk.sample_rate,
                            final=bool(chunk.is_final),
                        )
        finally:
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)

    def _build_inputs(
        self, request: SynthesisRequest, reference_path: Path | None
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": request.request_id,
            "text": request.text,
            "instruction": request.instruction or "Speak clearly and naturally.",
            "speaker": request.voice or "S0",
        }
        template_name = "tts_instruction"
        if reference_path is not None:
            if not request.reference_text:
                raise ValueError("Breeze reference audio requires reference_text")
            item["ref_audio_path"] = str(reference_path)
            item["ref_text"] = request.reference_text
            template_name = "ref_edit_tata"
        return self._prepare_inputs(
            self._tokenizer,
            self._audio_tokenizer,
            self._model,
            [item],
            self._get_template(template_name),
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

    @staticmethod
    def _float_to_pcm16(audio: Any) -> bytes:
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32)
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767.0).astype("<i2", copy=False).tobytes()

    async def close(self) -> None:
        runtime = self._runtime
        close = getattr(runtime, "close", None)
        if close is not None:
            await asyncio.to_thread(close)
        self._runtime = None
        self._model = None
        self._tokenizer = None
        self._audio_tokenizer = None
