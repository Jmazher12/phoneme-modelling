import zlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from phoneme_modelling.util.acoustics import dsp, landmarks
from phoneme_modelling.util.acoustics.clip import Clip, Segment

SAMPLE_RATE = 16000
CLIP_SECONDS = 2.5
MIN_PAD_SECONDS = 0.3
SPEECH_MARGIN_MS = 50.0
EDGE_FADE_MS = 10.0
LIGHT_NOISE_DB = -60.0
BACKGROUND_FFT = 1024
OUTPUT_PEAK = 10 ** (-3 / 20)
EXTENT_FLOOR_DB = 10.0
EXTENT_RANGE_DB = -50.0
EXTENT_WINDOW_MS = 300.0


def stable_rng(*parts: object) -> np.random.Generator:
    """Random generator seeded from ``parts``, so the same input always gives the same audio."""
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def speech_extent(samples: np.ndarray, rate: int) -> tuple[int, int]:
    """Main active span, widened to any nearby frame clearly above the noise floor (weak prevoicing, decaying tails)."""
    start, end = landmarks.active_span(samples, rate)
    db = landmarks.frame_db(samples, rate, 10.0)
    hop = dsp.to_samples(rate, 10.0)
    active = np.flatnonzero(db > max(float(np.percentile(db, 10)) + EXTENT_FLOOR_DB, EXTENT_RANGE_DB))
    window = dsp.to_samples(rate, EXTENT_WINDOW_MS)
    active = active[(active * hop >= start - window) & (active * hop <= end + window)]
    if len(active):
        start, end = min(start, int(active[0] * hop)), max(end, int((active[-1] + 1) * hop))
    return start, min(end, len(samples))


def as_clip(samples: np.ndarray) -> Clip:
    start, end = speech_extent(samples, SAMPLE_RATE)
    return Clip(samples, SAMPLE_RATE, [Segment("x", start / SAMPLE_RATE, end / SAMPLE_RATE)])


def background_noise(seed: Clip, length: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary noise with the average spectrum and level of ``seed``'s non-speech parts.

    Tiling a short background snippet repeats it at a fixed rate, which a model hears as phones ("bəbəb"), so the
    background is resynthesized with random phase instead. Seeds trimmed to digital silence get light white noise
    LIGHT_NOISE_DB below their peak.
    """
    rate = seed.sample_rate
    guard = dsp.to_samples(rate, 20)
    first, last = seed.speech_bounds()
    background = np.concatenate([seed.samples[: max(0, first - guard)], seed.samples[last + guard :]])
    floor = float(np.max(np.abs(seed.samples), initial=0.0)) * 10 ** (LIGHT_NOISE_DB / 20)
    if len(background) < dsp.to_samples(rate, 50) or dsp.rms(background) < floor:
        return rng.standard_normal(length) * floor

    hop = BACKGROUND_FFT // 4
    magnitude = np.abs(librosa.stft(background, n_fft=BACKGROUND_FFT, hop_length=hop)).mean(axis=1)
    phase = np.exp(2j * np.pi * rng.random((len(magnitude), 1 + length // hop)))
    noise = librosa.istft(magnitude[:, None] * phase, hop_length=hop, length=length)
    return noise * (dsp.rms(background) / (dsp.rms(noise) + 1e-12))


def deliver(samples: np.ndarray, background: Clip, rng: np.random.Generator) -> np.ndarray:
    """Place the speech of ``samples`` in the delivery format: 16 kHz, >= 2.5 s, at a random offset inside noise
    resynthesized from ``background``. Clips too long for 2.5 s are lengthened rather than cut."""
    clip = as_clip(samples)
    rate = clip.sample_rate
    margin = dsp.to_samples(rate, SPEECH_MARGIN_MS)
    first, last = clip.speech_bounds()
    speech = clip.samples[max(0, first - margin) : min(len(clip.samples), last + margin)]

    pad = dsp.to_samples(rate, MIN_PAD_SECONDS * 1000)
    length = max(dsp.to_samples(rate, CLIP_SECONDS * 1000), len(speech) + 2 * pad)
    output = background_noise(background, length, rng)
    offset = int(rng.integers(pad, length - len(speech) - pad + 1))

    fade = min(dsp.to_samples(rate, EDGE_FADE_MS), len(speech) // 2)
    weight = dsp.envelope(len(speech), fade, fade)
    region = output[offset : offset + len(speech)]
    output[offset : offset + len(speech)] = region * (1 - weight) + speech * weight
    return output


def save_wav(path: Path, samples: np.ndarray) -> None:
    """Write 16-bit PCM WAV, scaling down only if the clip would clip."""
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 0.999:
        samples = samples * (0.999 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
