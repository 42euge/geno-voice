import os
import tempfile
import time

from .base import STTEngine


class WhisperEngine(STTEngine):
    name = "whisper"

    def __init__(self, model_repo: str = "mlx-community/whisper-large-v3-turbo", **kwargs):
        self.model_repo = model_repo
        self._mlx_whisper = None

    def _load(self):
        if self._mlx_whisper is None:
            import mlx_whisper
            self._mlx_whisper = mlx_whisper

    def transcribe(self, wav_bytes: bytes) -> tuple[str | None, float]:
        self._load()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        try:
            t0 = time.monotonic()
            result = self._mlx_whisper.transcribe(
                tmp_path, path_or_hf_repo=self.model_repo
            )
            elapsed = time.monotonic() - t0
            return result["text"].strip(), elapsed
        except Exception:
            return None, time.monotonic() - t0
        finally:
            os.unlink(tmp_path)
