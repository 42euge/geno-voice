"""
Arousal tracker — real-time emotional activation from audio features.

Extracts pitch (F0), energy (RMS), and vocal tension (ZCR) from audio
chunks, normalizes against a per-session speaker baseline, and computes
a running arousal score (0.0 = calm, 1.0 = agitated) with dual-rate
EMA trajectory tracking.

No ML dependencies — pure numpy, <2ms per chunk.

Feeds into:
  - Turn-taking engine (emotional_content_recent, user_crying flags)
  - Canvas visualization (particle speed, density, turbulence)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ArousalState:
    score: float = 0.5
    fast_ema: float = 0.5
    slow_ema: float = 0.5
    trajectory: float = 0.0
    is_elevated: bool = False
    is_crying: bool = False
    chunks_processed: int = 0


@dataclass
class AudioFeatures:
    rms: float = 0.0
    energy_var: float = 0.0
    f0_hz: float = 0.0
    f0_confidence: float = 0.0
    f0_std: float = 0.0
    zcr: float = 0.0


class ArousalTracker:
    def __init__(
        self,
        sample_rate: int = 16000,
        baseline_chunks: int = 5,
        fast_alpha: float = 0.3,
        slow_alpha: float = 0.05,
        elevated_threshold: float = 0.65,
        crying_jitter_threshold: float = 0.15,
    ):
        self.sample_rate = sample_rate
        self.baseline_chunks = baseline_chunks
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.elevated_threshold = elevated_threshold
        self.crying_jitter_threshold = crying_jitter_threshold

        self._baseline_samples: list[AudioFeatures] = []
        self._baseline_ready = False
        self._baseline = {
            "rms_mean": 0.0, "rms_std": 1.0,
            "f0_mean": 0.0, "f0_std": 1.0,
            "f0var_mean": 0.0, "f0var_std": 1.0,
            "zcr_mean": 0.0, "zcr_std": 1.0,
        }
        self.state = ArousalState()

    def process_chunk(self, pcm_bytes: bytes) -> ArousalState:
        features = self.extract_features(pcm_bytes)

        if not self._baseline_ready:
            self._baseline_samples.append(features)
            if len(self._baseline_samples) >= self.baseline_chunks:
                self._compute_baseline()
            return self.state

        arousal = self._compute_arousal(features)
        crying = self._detect_crying(features)

        self.state.score = arousal
        self.state.fast_ema = self.fast_alpha * arousal + (1 - self.fast_alpha) * self.state.fast_ema
        self.state.slow_ema = self.slow_alpha * arousal + (1 - self.slow_alpha) * self.state.slow_ema
        self.state.trajectory = self.state.fast_ema - self.state.slow_ema
        self.state.is_elevated = self.state.fast_ema > self.elevated_threshold
        self.state.is_crying = crying
        self.state.chunks_processed += 1

        return self.state

    def extract_features(self, pcm_bytes: bytes) -> AudioFeatures:
        n_samples = len(pcm_bytes) // 2
        if n_samples == 0:
            return AudioFeatures()

        samples = np.frombuffer(pcm_bytes[:n_samples * 2], dtype=np.int16).astype(np.float32) / 32768.0

        rms = float(np.sqrt(np.mean(samples ** 2)))

        win_size = self.sample_rate // 10
        n_windows = len(samples) // win_size
        if n_windows > 1:
            windowed_rms = np.array([
                np.sqrt(np.mean(samples[i * win_size:(i + 1) * win_size] ** 2))
                for i in range(n_windows)
            ])
            energy_var = float(np.std(windowed_rms))
        else:
            energy_var = 0.0

        f0, f0_conf = self._estimate_f0(samples)

        f0_values = self._windowed_f0(samples, window_ms=50)
        voiced = f0_values[f0_values > 0]
        f0_std = float(np.std(voiced)) if len(voiced) > 1 else 0.0

        zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2)

        return AudioFeatures(
            rms=rms, energy_var=energy_var,
            f0_hz=f0, f0_confidence=f0_conf, f0_std=f0_std,
            zcr=zcr,
        )

    def _estimate_f0(self, samples: np.ndarray) -> tuple[float, float]:
        fmin, fmax = 70.0, 400.0
        min_lag = int(self.sample_rate / fmax)
        max_lag = int(self.sample_rate / fmin)

        if len(samples) < max_lag * 2:
            return 0.0, 0.0

        corr = np.correlate(samples, samples, mode='full')
        corr = corr[len(corr) // 2:]
        corr = corr / (corr[0] + 1e-10)

        search = corr[min_lag:max_lag]
        if len(search) == 0:
            return 0.0, 0.0

        peak_idx = int(np.argmax(search))
        confidence = float(search[peak_idx])
        lag = peak_idx + min_lag
        f0 = self.sample_rate / lag if lag > 0 else 0.0

        return f0, confidence

    def _windowed_f0(self, samples: np.ndarray, window_ms: int = 50) -> np.ndarray:
        win_size = int(self.sample_rate * window_ms / 1000)
        hop = win_size // 2
        f0s = []
        for start in range(0, len(samples) - win_size, hop):
            f0, conf = self._estimate_f0(samples[start:start + win_size])
            f0s.append(f0 if conf > 0.3 else 0.0)
        return np.array(f0s) if f0s else np.array([0.0])

    def _compute_baseline(self):
        rms_vals = [f.rms for f in self._baseline_samples]
        f0_vals = [f.f0_hz for f in self._baseline_samples if f.f0_hz > 0]
        f0var_vals = [f.f0_std for f in self._baseline_samples]
        zcr_vals = [f.zcr for f in self._baseline_samples]

        self._baseline = {
            "rms_mean": np.mean(rms_vals), "rms_std": max(np.std(rms_vals), 1e-6),
            "f0_mean": np.mean(f0_vals) if f0_vals else 150.0,
            "f0_std": max(np.std(f0_vals), 1e-6) if f0_vals else 30.0,
            "f0var_mean": np.mean(f0var_vals), "f0var_std": max(np.std(f0var_vals), 1e-6),
            "zcr_mean": np.mean(zcr_vals), "zcr_std": max(np.std(zcr_vals), 1e-6),
        }
        self._baseline_ready = True

    def _compute_arousal(self, f: AudioFeatures) -> float:
        b = self._baseline
        rms_z = self._z(f.rms, b["rms_mean"], b["rms_std"])
        f0_z = self._z(f.f0_hz, b["f0_mean"], b["f0_std"]) if f.f0_hz > 0 else 0.0
        f0var_z = self._z(f.f0_std, b["f0var_mean"], b["f0var_std"])
        zcr_z = self._z(f.zcr, b["zcr_mean"], b["zcr_std"])

        raw = (
            0.30 * rms_z +
            0.25 * f0_z +
            0.25 * f0var_z +
            0.10 * zcr_z +
            0.10 * min(f.energy_var * 10, 3.0)
        )
        raw = np.clip(raw, -6, 6)
        return float(1.0 / (1.0 + np.exp(-raw)))

    def _detect_crying(self, f: AudioFeatures) -> bool:
        if f.f0_hz <= 0:
            return False
        high_f0_var = f.f0_std > (self._baseline["f0var_mean"] + 2 * self._baseline["f0var_std"])
        high_energy_var = f.energy_var > 0.05
        return high_f0_var and high_energy_var

    @staticmethod
    def _z(value: float, mean: float, std: float, max_z: float = 3.0) -> float:
        if std < 1e-6:
            return 0.0
        return float(np.clip((value - mean) / std, -max_z, max_z))
