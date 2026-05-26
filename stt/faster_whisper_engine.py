"""Faster-Whisper STT engine — x86_64 Linux STT path.

iter-118: parallel implementation to ``WhisperEngine`` (MLX,
Apple Silicon only). Implements the same ``STTEngine`` contract
so the mic_chat.py factory closure (iter-108's ``stt_factory``)
can drop in either backend without touching the loop.

Lazy-imports ``faster_whisper`` inside ``_load()`` so the module
itself stays importable on systems where the package isn't
installed (matches iter-109 GENO.md rule 5: "lazy-import
platform deps inside the closures, not at module scope").

Model-repo string semantics:
  - Short aliases ("tiny", "base", "small", "medium", "large-v3",
    etc.) — passed through to faster-whisper, which resolves
    against the Hugging Face hub (or local cache).
  - Full repo IDs ("Systran/faster-whisper-large-v3") — also
    pass through.
  - MLX-style repos ("mlx-community/whisper-large-v3-turbo") —
    best-effort stripped to the bare size string. The conversion
    is a heuristic; users who want exact control should pass
    a faster-whisper-native repo string.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time

from .base import STTEngine

log = logging.getLogger("geno-voice.stt.faster_whisper")


# Maps an MLX-style repo string ("mlx-community/whisper-large-v3-turbo")
# to a faster-whisper-recognizable size token. Best-effort —
# users wanting precise control should pass a faster-whisper-
# native string directly.
_MLX_REPO_PATTERN = re.compile(
    r"^mlx-community/whisper-(?P<size>tiny|base|small|medium|large(?:-v\d+)?(?:-turbo)?)$"
)


def _resolve_model_repo(model_repo: str) -> str:
    """Translate a model-repo string into a faster-whisper-
    recognizable form.

    Short aliases pass through. MLX-community repos lose the
    namespace + "whisper-" prefix. Other strings (third-party
    HF repos, local paths) pass through unchanged.
    """
    m = _MLX_REPO_PATTERN.match(model_repo)
    if m:
        return m.group("size")
    return model_repo


class FasterWhisperEngine(STTEngine):
    """STT engine backed by ``faster-whisper`` (CTranslate2 +
    quantized weights — runs on CPU or GPU).

    Defaults match the iter-117 audio-fixture integration test:
    ``tiny`` model, CPU device, ``int8`` compute type. Operators
    can override via constructor kwargs:

        FasterWhisperEngine(
            model_repo="large-v3",
            device="cuda",
            compute_type="float16",
        )

    Class-level model cache: a single ``faster_whisper.WhisperModel``
    instance per (model_repo, device, compute_type) tuple is shared
    across instances. Mirrors WhisperEngine's class-level cache —
    avoids re-loading the model on every ChatLoop construction.
    """

    name = "faster_whisper"

    # Class-level cache: (resolved_repo, device, compute_type) → WhisperModel.
    _model_cache: dict = {}
    _load_lock = threading.Lock()

    def __init__(
        self,
        model_repo: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        **kwargs,
    ):
        self.model_repo = model_repo
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    def _cache_key(self) -> tuple:
        return (
            _resolve_model_repo(self.model_repo),
            self.device,
            self.compute_type,
        )

    def _load(self):
        key = self._cache_key()
        cached = type(self)._model_cache.get(key)
        if cached is not None:
            self._model = cached
            return

        with type(self)._load_lock:
            cached = type(self)._model_cache.get(key)
            if cached is not None:
                self._model = cached
                return
            # Lazy import so missing faster-whisper at module level
            # doesn't break unrelated callers.
            from faster_whisper import WhisperModel
            resolved = _resolve_model_repo(self.model_repo)
            self._model = WhisperModel(
                resolved, device=self.device,
                compute_type=self.compute_type,
            )
            type(self)._model_cache[key] = self._model

    def transcribe(self, wav_bytes: bytes) -> tuple[str | None, float]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        t0 = time.monotonic()
        try:
            with self._lock:
                self._load()
                segments, _info = self._model.transcribe(
                    tmp_path, language="en",
                )
                # Generator — must consume immediately so the
                # underlying file is read before the temp file is
                # unlinked.
                text = " ".join(s.text for s in segments).strip()
            elapsed = time.monotonic() - t0
            return text or None, elapsed
        except Exception as exc:
            log.exception("FasterWhisper transcription failed: %s", exc)
            return None, time.monotonic() - t0
        finally:
            os.unlink(tmp_path)
