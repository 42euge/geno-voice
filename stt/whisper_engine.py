import logging
import os
import tempfile
import threading
import time

from .base import STTEngine

log = logging.getLogger("geno-voice.stt.whisper")


class WhisperEngine(STTEngine):
    name = "whisper"
    _shared_mlx_whisper = None
    _loaded_model_repo = None
    _load_lock = threading.Lock()

    def __init__(
        self, model_repo: str = "mlx-community/whisper-large-v3-turbo", **kwargs
    ):
        self.model_repo = model_repo
        self._mlx_whisper = None
        self._lock = threading.Lock()

    def _load(self):
        cls = type(self)
        if cls._loaded_model_repo == self.model_repo and cls._shared_mlx_whisper is not None:
            self._mlx_whisper = cls._shared_mlx_whisper
            return

        with cls._load_lock:
            if cls._shared_mlx_whisper is None:
                import mlx_whisper

                cls._shared_mlx_whisper = mlx_whisper
            self._mlx_whisper = cls._shared_mlx_whisper

            if cls._loaded_model_repo == self.model_repo:
                return

            import mlx.core as mx
            import mlx_whisper
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.get_model(self.model_repo, dtype=mx.float16)
            cls._loaded_model_repo = self.model_repo

    def transcribe(self, wav_bytes: bytes) -> tuple[str | None, float]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        t0 = time.monotonic()
        try:
            # MLX Whisper model loading / inference is not reliably thread-safe
            # during cold start. Serialize access so overlapping microphone
            # requests do not crash the process.
            with self._lock:
                self._load()
                result = self._mlx_whisper.transcribe(
                    tmp_path, path_or_hf_repo=self.model_repo
                )
            elapsed = time.monotonic() - t0
            return result["text"].strip(), elapsed
        except Exception as exc:
            log.exception("Whisper transcription failed: %s", exc)
            return None, time.monotonic() - t0
        finally:
            os.unlink(tmp_path)
