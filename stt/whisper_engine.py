import logging
import os
import tempfile
import threading
import time

from .base import STTEngine

<<<<<<< Updated upstream
log = logging.getLogger("stt.whisper")
=======
log = logging.getLogger("geno-voice.stt.whisper")
>>>>>>> Stashed changes


class WhisperEngine(STTEngine):
    name = "whisper"

    def __init__(self, model_repo: str = "mlx-community/whisper-large-v3-turbo", **kwargs):
        self.model_repo = model_repo
        self._mlx_whisper = None
        self._lock = threading.Lock()

    def _load(self):
        if self._mlx_whisper is None:
            import mlx_whisper
            self._mlx_whisper = mlx_whisper

    def transcribe(self, wav_bytes: bytes) -> tuple[str | None, float]:
<<<<<<< Updated upstream
        if len(wav_bytes) < 1000:
            return "", 0.0

        self._load()
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        except Exception as e:
            log.error("transcription failed: %s", e)
=======
        except Exception as exc:
            log.exception("Whisper transcription failed: %s", exc)
>>>>>>> Stashed changes
            return None, time.monotonic() - t0
        finally:
            os.unlink(tmp_path)
