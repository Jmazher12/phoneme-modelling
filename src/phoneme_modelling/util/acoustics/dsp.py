from typing import Optional

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import butter, sosfiltfilt

from phoneme_modelling.util.acoustics.clip import Clip, Segment

FADE_MS = 8.0
HIGH_VOICE_F0 = 165.0
MIN_FORMANT_HZ = 100.0


def to_samples(sample_rate: int, milliseconds: float) -> int:
    return int(round(sample_rate * milliseconds / 1000))


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0


def ramp(length: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, length)) if length > 0 else np.zeros(0)


def envelope(length: int, rise: int, fall: int) -> np.ndarray:
    shape = np.ones(length)
    rise, fall = min(rise, length // 2), min(fall, length // 2)
    shape[:rise] = ramp(rise)
    if fall:
        shape[-fall:] = ramp(fall)[::-1]
    return shape


def fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) >= length:
        return samples[:length]
    return np.pad(samples, (0, length - len(samples)))


def lowpass(samples: np.ndarray, sample_rate: int, cutoff: float, order: int = 6) -> np.ndarray:
    if len(samples) < 3 * order * 2:
        return samples.copy()
    sos = butter(order, cutoff, btype="low", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, samples)


def highpass(samples: np.ndarray, sample_rate: int, cutoff: float, order: int = 6) -> np.ndarray:
    if len(samples) < 3 * order * 2:
        return np.zeros_like(samples)
    sos = butter(order, cutoff, btype="high", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, samples)


def median_f0(samples: np.ndarray, sample_rate: int) -> Optional[float]:
    if len(samples) < to_samples(sample_rate, 40):
        return None
    pitch = parselmouth.Sound(samples, sample_rate).to_pitch(pitch_floor=75.0, pitch_ceiling=600.0)
    voiced = pitch.selected_array["frequency"]
    voiced = voiced[voiced > 0]
    return float(np.median(voiced)) if len(voiced) else None


def max_formant_for(clip: Clip) -> float:
    """Praat's usual formant ceiling: 5000 Hz for low voices, 5500 Hz for higher voices."""
    f0 = median_f0(clip.samples, clip.sample_rate)
    return 5500.0 if f0 is not None and f0 > HIGH_VOICE_F0 else 5000.0


def _map_time(time: float, start: float, end: float, new_length: float) -> float:
    if time <= start:
        return time
    if time >= end:
        return time + new_length - (end - start)
    return start + (time - start) * (new_length / (end - start))


def replace_span(clip: Clip, start: int, end: int, new: np.ndarray, fade_ms: float = FADE_MS) -> Clip:
    """Replace ``samples[start:end]`` by ``new``; the new audio fades in over the old signal at both joints."""
    samples = clip.samples
    new = np.array(new, dtype=np.float64, copy=True)
    fade = min(to_samples(clip.sample_rate, fade_ms), len(new) // 2)
    if fade > 0:
        fade_in = ramp(fade)
        if start > 0:
            old = np.zeros(fade)
            available = samples[start : start + fade]
            old[: len(available)] = available
            new[:fade] = new[:fade] * fade_in + old * (1 - fade_in)
        if end < len(samples):
            old = np.zeros(fade)
            available = samples[max(start, end - fade) : end]
            old[fade - len(available) :] = available
            new[-fade:] = new[-fade:] * fade_in[::-1] + old * (1 - fade_in[::-1])
    output = np.concatenate([samples[:start], new, samples[end:]])

    rate = clip.sample_rate
    start_t, end_t, new_t = start / rate, end / rate, len(new) / rate
    segments = [
        Segment(
            segment.phone,
            _map_time(segment.start, start_t, end_t, new_t),
            _map_time(segment.end, start_t, end_t, new_t),
        )
        for segment in clip.segments
    ]
    return clip.with_samples(output, [segment for segment in segments if segment.end - segment.start > 1e-4])


def shift_formants(clip: Clip, index: int, targets: dict[int, float], alpha: float, ramp_ms: float = 40.0) -> Clip:
    """Pull formant tracks of segment ``index`` a fraction ``alpha`` of the way toward ``targets``.

    The weight is 1 inside the segment and ramps to 0 over ``ramp_ms`` into its neighbors, so transitions into a
    vowel move with the consonant. Only the band below the formant ceiling is resynthesized; higher frequencies
    are kept from the original so the clip does not sound muffled.
    """
    rate = clip.sample_rate
    segment = clip.segments[index]
    ramp_s = ramp_ms / 1000
    max_formant = max_formant_for(clip)
    analysis_rate = 2 * max_formant

    sound = parselmouth.Sound(clip.samples, rate)
    resampled = call(sound, "Resample", analysis_rate, 50)
    lpc = call(resampled, "To LPC (burg)", 12, 0.025, 0.005, 50)
    source = call([resampled, lpc], "Filter (inverse)")
    formant = call(resampled, "To Formant (burg)", 0.005, 5, max_formant, 0.025, 50)
    grid = call(formant, "Down to FormantGrid")

    region_start = max(0.0, segment.start - ramp_s)
    region_end = min(clip.duration, segment.end + ramp_s)
    times = np.arange(region_start, region_end, 0.005)
    for number, target in targets.items():
        values = [call(formant, "Get value at time", number, time, "hertz", "linear") for time in times]
        call(grid, "Remove formant points between", number, region_start, region_end)
        for time, value in zip(times, values, strict=True):
            if not np.isfinite(value):
                continue
            weight = min(1.0, max(0.0, (time - (segment.start - ramp_s)) / ramp_s))
            weight *= min(1.0, max(0.0, ((segment.end + ramp_s) - time) / ramp_s))
            shifted = max(MIN_FORMANT_HZ, value + alpha * weight * (target - value))  # alpha > 1 can overshoot
            call(grid, "Add formant point", number, time, shifted)

    filtered = call([source, grid], "Filter").values[0]
    filtered = fit_length(librosa.resample(filtered, orig_sr=analysis_rate, target_sr=rate), len(clip.samples))

    pad = to_samples(rate, 10)
    start = max(0, int(region_start * rate) - pad)
    end = min(len(clip.samples), int(region_end * rate) + pad)
    cutoff = 0.9 * max_formant
    original = clip.samples[start:end]
    new_low = lowpass(filtered[start:end], rate, cutoff)
    new_low *= rms(lowpass(original, rate, cutoff)) / (rms(new_low) + 1e-12)
    region = new_low + highpass(original, rate, cutoff)
    return replace_span(clip, start, end, region, fade_ms=10.0)
