"""
Extracts F0 (pitch), Energy (RMS), Jitter, Shimmer, and HNR from audio
using parselmouth (Praat bindings).
"""

from dataclasses import dataclass

import numpy as np
import parselmouth
from parselmouth.praat import call


@dataclass
class ProsodicFeatures:
    f0_mean: float          # Mean fundamental frequency (Hz)
    f0_std: float           # F0 standard deviation
    energy_mean: float      # Mean RMS energy
    energy_std: float       # RMS energy standard deviation
    jitter_local: float     # Local jitter (period-to-period variation)
    shimmer_local: float    # Local shimmer (amplitude variation)
    hnr_mean: float         # Mean Harmonics-to-Noise Ratio (dB)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.f0_mean,
            self.energy_mean,
            self.jitter_local,
            self.shimmer_local,
            self.hnr_mean,
        ], dtype=np.float32)

    def to_full_array(self) -> np.ndarray:
        return np.array([
            self.f0_mean, self.f0_std,
            self.energy_mean, self.energy_std,
            self.jitter_local,
            self.shimmer_local,
            self.hnr_mean,
        ], dtype=np.float32)


def extract_prosodic_features(
    audio_path: str,
    f0_min: float = 75.0,
    f0_max: float = 500.0,
) -> ProsodicFeatures:
    snd = parselmouth.Sound(audio_path)

    # --- F0 (Pitch) ---
    pitch = snd.to_pitch(pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0_values = pitch.selected_array["frequency"]
    f0_voiced = f0_values[f0_values > 0]  # only voiced frames
    f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0

    # --- Energy (RMS) ---
    intensity = snd.to_intensity()
    energy_values = intensity.values[0]
    energy_mean = float(np.mean(energy_values)) if len(energy_values) > 0 else 0.0
    energy_std = float(np.std(energy_values)) if len(energy_values) > 0 else 0.0

    # --- Jitter (local) ---
    point_process = call(snd, "To PointProcess (periodic, cc)",
                         f0_min, f0_max)
    try:
        jitter_local = call(point_process, "Get jitter (local)",
                            0.0, 0.0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter_local = 0.0
    if np.isnan(jitter_local):
        jitter_local = 0.0

    # --- Shimmer (local) ---
    try:
        shimmer_local = call(
            [snd, point_process], "Get shimmer (local)",
            0.0, 0.0, 0.0001, 0.02, 1.3, 1.6,
        )
    except Exception:
        shimmer_local = 0.0
    if np.isnan(shimmer_local):
        shimmer_local = 0.0

    # --- HNR (Harmonics-to-Noise Ratio) ---
    harmonicity = call(snd, "To Harmonicity (cc)",
                       0.01, f0_min, 0.1, 1.0)
    hnr_values = call(harmonicity, "Get mean", 0.0, 0.0)
    if np.isnan(hnr_values):
        hnr_values = 0.0
    hnr_mean = float(hnr_values)

    return ProsodicFeatures(
        f0_mean=f0_mean,
        f0_std=f0_std,
        energy_mean=energy_mean,
        energy_std=energy_std,
        jitter_local=jitter_local,
        shimmer_local=shimmer_local,
        hnr_mean=hnr_mean,
    )


def extract_batch(
    audio_paths: list[str],
    f0_min: float = 75.0,
    f0_max: float = 500.0,
) -> np.ndarray:
    features = []
    for path in audio_paths:
        try:
            pf = extract_prosodic_features(path, f0_min, f0_max)
            features.append(pf.to_array())
        except Exception:
            # Return zeros for failed extractions
            features.append(np.zeros(5, dtype=np.float32))
    return np.stack(features)
