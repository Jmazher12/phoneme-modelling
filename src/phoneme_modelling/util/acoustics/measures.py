from dataclasses import dataclass

import numpy as np
import parselmouth

from phoneme_modelling.phonemes import TEXT_TO_IPA
from phoneme_modelling.util.acoustics import dsp, landmarks
from phoneme_modelling.util.acoustics.clip import Clip, Segment

NAN = float("nan")
PRIMARY_BURST_MS = 20.0
VOICEBAR_BAND = (50.0, 400.0)
# The sources are adult voices (F0 76-250 Hz); a 600 Hz ceiling picked up "F0" of 450-540 Hz in voiceless aspiration.
F0_CEILING_HZ = 400.0
MIN_VOICED_SHARE_FOR_F0 = 0.25


@dataclass(frozen=True)
class SoundInfo:
    manner: str  # stop, affricate, fricative, nasal, approximant
    voiced: bool


SOUNDS: dict[str, SoundInfo] = {
    "b": SoundInfo("stop", True),
    "p": SoundInfo("stop", False),
    "d": SoundInfo("stop", True),
    "t": SoundInfo("stop", False),
    "g": SoundInfo("stop", True),
    "k": SoundInfo("stop", False),
    "ch": SoundInfo("affricate", False),
    "j": SoundInfo("affricate", True),
    "f": SoundInfo("fricative", False),
    "v": SoundInfo("fricative", True),
    "th_voiceless": SoundInfo("fricative", False),
    "th_voiced": SoundInfo("fricative", True),
    "s": SoundInfo("fricative", False),
    "z": SoundInfo("fricative", True),
    "sh": SoundInfo("fricative", False),
    "h": SoundInfo("fricative", False),
    "m": SoundInfo("nasal", True),
    "n": SoundInfo("nasal", True),
    "ng": SoundInfo("nasal", True),
    "r": SoundInfo("approximant", True),
    "l": SoundInfo("approximant", True),
    "w": SoundInfo("approximant", True),
    "y": SoundInfo("approximant", True),
}

# Voiceless sibilant-like fricatives get the frication-span rule; weak or voiced fricatives and /h/ use the active span.
FRICATION_RULE = frozenset({"s", "sh", "f", "th_voiceless"})

NUISANCE_MEASURES = ("duration_ms", "f0_hz", "rms_dbfs")
CUE_MEASURES = (
    "voiced_fraction",
    "hnr_db",
    "voicebar_db",
    "cog_hz",
    "sd_hz",
    "skewness",
    "kurtosis",
    "peak_hz",
    "rise_ms",
    "onset_rise_db",
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "vot_ms",
    "closure_voicing_ms",
    "release_voicing",
    "burst_cog_hz",
    "closure_level_db",
    "aspiration_db",
)
# Compared as log ratios (proportional change); everything else as a difference.
RATIO_MEASURES = frozenset({"cog_hz", "peak_hz", "f1_hz", "f2_hz", "f3_hz", "burst_cog_hz", "duration_ms", "f0_hz"})


def segment_sound(samples: np.ndarray, rate: int, key: str) -> Clip:
    """Clip whose first segment is the target sound (a following vowel, if any, is segment 1)."""
    info = SOUNDS[key]
    ipa = TEXT_TO_IPA[key]
    if info.manner == "stop":
        segments = landmarks.segment_stop(samples, rate, ipa)
    elif key in FRICATION_RULE:
        segments = landmarks.segment_fricative(samples, rate, ipa)
    else:
        start, end = landmarks.active_span(samples, rate)
        segments = [Segment(ipa, start / rate, end / rate)]
    return Clip(samples, rate, segments)


def _voicing_in(clip: Clip, start: int, end: int) -> tuple[float, float]:
    """(voiced share, median F0) of 5 ms Praat frames between two sample indices."""
    if len(clip.samples) < dsp.to_samples(clip.sample_rate, 60):
        return 0.0, NAN
    pitch = parselmouth.Sound(clip.samples, clip.sample_rate).to_pitch_ac(
        time_step=0.005, pitch_floor=75.0, pitch_ceiling=F0_CEILING_HZ
    )
    times, frequencies = np.asarray(pitch.xs()), pitch.selected_array["frequency"]
    inside = (times >= start / clip.sample_rate) & (times < end / clip.sample_rate)
    if not inside.any():
        return 0.0, NAN
    voiced = frequencies[inside] > 0
    if np.mean(voiced) < MIN_VOICED_SHARE_FOR_F0:
        return float(np.mean(voiced)), NAN
    return float(np.mean(voiced)), float(np.median(frequencies[inside][voiced]))


def _hnr(clip: Clip, start: int, end: int) -> float:
    if end - start < dsp.to_samples(clip.sample_rate, 40):
        return NAN
    harmonicity = parselmouth.Sound(clip.samples[start:end], clip.sample_rate).to_harmonicity_cc(
        time_step=0.01, minimum_pitch=75.0
    )
    values = harmonicity.values[0]
    values = values[values > -199]
    return float(np.mean(values)) if len(values) else NAN


def _voicebar(freqs: np.ndarray, power: np.ndarray) -> float:
    low = power[(freqs >= VOICEBAR_BAND[0]) & (freqs < VOICEBAR_BAND[1])].sum()
    high = power[(freqs >= landmarks.MEASURE_BAND[0]) & (freqs <= landmarks.MEASURE_BAND[1])].sum()
    return float(10 * np.log10((low + 1e-20) / (high + 1e-20)))


def _rise_ms(samples: np.ndarray, rate: int) -> float:
    """Time for the 2 ms RMS envelope to climb from 10% to 90% of its maximum."""
    hop = dsp.to_samples(rate, 2.0)
    frames = len(samples) // hop
    if frames < 3:
        return NAN
    envelope = np.array([dsp.rms(samples[i * hop : (i + 1) * hop]) for i in range(frames)])
    envelope = np.convolve(envelope, np.ones(3) / 3, mode="same")
    top = envelope.max()
    first10 = int(np.flatnonzero(envelope >= 0.1 * top)[0])
    first90 = int(np.flatnonzero(envelope >= 0.9 * top)[0])
    return 2.0 * max(0, first90 - first10)


def _onset_rise_db(samples: np.ndarray, rate: int) -> float:
    """Steepest rise of > 2 kHz energy (dB per 1 ms frame, 3 ms smoothing): large for bursts, small for hiss onsets."""
    high = dsp.highpass(samples, rate, 2000.0, order=4)
    hop = dsp.to_samples(rate, 1.0)
    frames = len(high) // hop
    if frames < 5:
        return NAN
    energy = 10 * np.log10(np.array([np.sum(high[i * hop : (i + 1) * hop] ** 2) for i in range(frames)]) + 1e-20)
    energy = np.maximum(energy, energy.max() - landmarks.BURST_FLOOR_DB)
    energy = np.convolve(np.pad(energy, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    return float(np.max(np.diff(energy)))


def _formants(clip: Clip, start: int, end: int) -> dict[str, float]:
    rate = clip.sample_rate
    quarter = (end - start) // 4
    t0, t1 = (start + quarter) / rate, (end - quarter) / rate
    if t1 - t0 < 0.015:
        return {"f1_hz": NAN, "f2_hz": NAN, "f3_hz": NAN}
    _, tracks = landmarks.formant_tracks(clip, t0, t1)
    return {f"f{n}_hz": float(np.nanmedian(tracks[n])) if np.isfinite(tracks[n]).any() else NAN for n in (1, 2, 3)}


def burst_window(clip: Clip) -> tuple[int, int, float]:
    """(start, end, burst time in s) of the first PRIMARY_BURST_MS after the release of segment 0."""
    rate = clip.sample_rate
    start, end = clip.bounds(0)
    burst = landmarks.burst_time(clip.samples, rate, start, min(end, start + dsp.to_samples(rate, 400.0)))
    first = int(burst * rate)
    return first, min(len(clip.samples), first + dsp.to_samples(rate, PRIMARY_BURST_MS)), burst


def measure_sound(clip: Clip, key: str) -> dict[str, object]:
    """All NUISANCE_MEASURES and CUE_MEASURES for segment 0 of ``clip`` (measured as sound ``key``).

    Every key is always present; a measure that does not apply (formants of /s/, burst of /m/) is NaN.
    """
    info = SOUNDS[key]
    rate = clip.sample_rate
    speech_start, speech_end = clip.speech_bounds()
    start, end = clip.bounds(0)
    target = clip.samples[start:end]
    values: dict[str, object] = dict.fromkeys(CUE_MEASURES + NUISANCE_MEASURES, NAN)

    # Nuisance measures use one rule for every sound (the active span), so a change of segmentation rule between
    # two sounds never shows up as a duration or level difference.
    active_start, active_end = landmarks.active_span(clip.samples, rate)
    values["duration_ms"] = 1000 * (active_end - active_start) / rate
    values["rms_dbfs"] = 20 * np.log10(dsp.rms(clip.samples[active_start:active_end]) + 1e-12)
    values["voiced_fraction"], values["f0_hz"] = _voicing_in(clip, speech_start, speech_end)
    values["hnr_db"] = _hnr(clip, start, end)
    values["rise_ms"] = _rise_ms(target, rate)
    values["onset_rise_db"] = _onset_rise_db(target, rate)

    trim = (end - start) // 10
    freqs, power = landmarks.power_spectrum(clip.samples[start + trim : end - trim], rate)
    values |= landmarks.spectral_moments(freqs, power)
    span_shape = landmarks.smoothed_db(freqs, power)
    values["peak_hz"] = float(landmarks.SHAPE_GRID[int(np.argmax(span_shape))])
    values["voicebar_db"] = _voicebar(freqs, power)

    if info.manner in ("stop", "affricate"):
        stop = landmarks.stop_measures(clip, 0)
        values |= stop
        first, _, _ = burst_window(clip)
        closure = clip.samples[start:first]
        burst_rms = dsp.rms(clip.samples[first : first + dsp.to_samples(rate, 10.0)]) + 1e-12
        if stop["closure_voicing_ms"] >= 10.0 and len(closure):
            values["closure_level_db"] = 20 * np.log10(dsp.rms(closure) / burst_rms)
        if stop["vot_ms"] >= 20.0:
            noise = clip.samples[first + dsp.to_samples(rate, 10.0) : first + dsp.to_samples(rate, stop["vot_ms"])]
            values["aspiration_db"] = 20 * np.log10(dsp.rms(noise) / burst_rms + 1e-12)
    if info.manner in ("nasal", "approximant") or (info.voiced and values["voiced_fraction"] > 0.3):
        values |= _formants(clip, start, end)
    return values


def measure_as(samples: np.ndarray, rate: int, key: str) -> dict[str, object]:
    return measure_sound(segment_sound(samples, rate, key), key)
