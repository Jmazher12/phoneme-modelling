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


def stop(rng: np.random.Generator, voiced: bool = False, frication_ms: float = 0.0) -> np.ndarray:
    """Closure, burst, release noise and a vowel. Voiced stops get a voice bar in the closure and a short release;
    ``frication_ms`` adds a long hiss after the burst (an affricate)."""
    burst = rng.standard_normal(seconds(0.008)) * np.exp(-np.linspace(0, 4, seconds(0.008)))
    closure = 1e-4 * rng.standard_normal(seconds(0.3))
    if voiced:
        closure[-seconds(0.06) :] += 0.05 * dsp.lowpass(voiced_sound(0.06), RATE, 500.0, order=4)
    release = hiss(0.015 if voiced else 0.05, 1000.0, rng, level=0.15)
    if frication_ms:
        release = hiss(frication_ms / 1000, 2500.0, rng, level=0.3)
    parts = [
        closure,
        0.8 * dsp.highpass(burst, RATE, 1500.0, order=2),
        release,
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
        "l": padded(voiced_sound(0.5, formants=(380.0, 1000.0, 2700.0)), rng),
        "w": padded(voiced_sound(0.5, formants=(300.0, 700.0, 2200.0)), rng),
        "y": padded(voiced_sound(0.5, formants=(280.0, 2200.0, 3000.0)), rng),
        "s": padded(hiss(0.6, 4000.0, rng), rng),
        "sh": padded(hiss(0.6, 2000.0, rng), rng),
        "f": padded(hiss(0.5, 1000.0, rng, level=0.1), rng),
        "th_voiceless": padded(hiss(0.5, 1500.0, rng, level=0.1), rng),
        "z": padded(voiced_sound(0.5, formants=(250.0, 1500.0, 2500.0)) * 0.3 + hiss(0.5, 4000.0, rng), rng),
        "v": padded(voiced_sound(0.4, formants=(300.0, 1200.0, 2400.0)) + hiss(0.4, 1000.0, rng, 0.05), rng),
        "th_voiced": padded(voiced_sound(0.4, formants=(300.0, 1400.0, 2600.0)) + hiss(0.4, 3000.0, rng, 0.05), rng),
        "m": padded(voiced_sound(0.5, formants=(250.0, 1000.0, 2200.0)), rng),
        "n": padded(voiced_sound(0.5, formants=(250.0, 1400.0, 2500.0)), rng),
        "ng": padded(voiced_sound(0.5, formants=(250.0, 1800.0, 2400.0)), rng),
        "p": stop(rng),
        "t": stop(rng),
        "k": stop(rng),
        "b": stop(rng, voiced=True),
        "d": stop(rng, voiced=True),
        "g": stop(rng, voiced=True),
        "ch": stop(rng, frication_ms=150.0),
        "j": stop(rng, voiced=True, frication_ms=120.0),
    }
    paths = {}
    for key, samples in sounds.items():
        paths[key] = tmp_path / f"000_{key}_kore.wav"
        sf.write(paths[key], samples, RATE, subtype="PCM_16")
    return paths
