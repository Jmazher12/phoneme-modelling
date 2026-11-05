from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import lfilter

from phoneme_modelling.util.acoustics import dsp

RATE = 16000


def seconds(value: float) -> int:
    return int(round(value * RATE))


def voiced_sound(
    duration: float, f0: float = 140.0, formants: tuple[float, ...] = (300.0, 1200.0, 2500.0)
) -> np.ndarray:
    source = np.zeros(seconds(duration))
    source[:: int(RATE / f0)] = 1.0
    for frequency in formants:
        radius = np.exp(-np.pi * 90.0 / RATE)
        theta = 2 * np.pi * frequency / RATE
        source = lfilter([1 - radius], [1, -2 * radius * np.cos(theta), radius**2], source)
    return 0.5 * source / np.max(np.abs(source))


def hiss(duration: float, low: float, rng: np.random.Generator, level: float = 0.3) -> np.ndarray:
    noise = dsp.highpass(rng.standard_normal(seconds(duration)), RATE, low)
    return level * noise / np.max(np.abs(noise))


def padded(sound: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Places a sound in quiet noise, like a real recording."""
    pad = 1e-4 * rng.standard_normal(seconds(0.4))
    return np.concatenate([pad, sound, pad])


def stop(rng: np.random.Generator) -> np.ndarray:
    burst = rng.standard_normal(seconds(0.008)) * np.exp(-np.linspace(0, 4, seconds(0.008)))
    parts = [
        1e-4 * rng.standard_normal(seconds(0.3)),
        0.8 * dsp.highpass(burst, RATE, 1500.0, order=2),
        hiss(0.05, 1000.0, rng, level=0.15),
        voiced_sound(0.08),
        1e-4 * rng.standard_normal(seconds(0.3)),
    ]
    return np.concatenate(parts)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def sound_files(tmp_path: Path, rng: np.random.Generator) -> dict[str, Path]:
    """A synthetic recording of each source sound, named the way the pipeline names clips."""
    sounds = {
        "r": padded(voiced_sound(0.5, formants=(400.0, 1100.0, 1700.0)), rng),
        "s": padded(hiss(0.6, 4000.0, rng), rng),
        "p": stop(rng),
        "th_voiced": padded(voiced_sound(0.4, formants=(300.0, 1400.0, 2600.0)) + hiss(0.4, 3000.0, rng, 0.05), rng),
        "m": padded(voiced_sound(0.5, formants=(250.0, 1000.0, 2200.0)), rng),
    }
    paths = {}
    for key, samples in sounds.items():
        paths[key] = tmp_path / f"000_{key}_kore.wav"
        sf.write(paths[key], samples, RATE, subtype="PCM_16")
    return paths
