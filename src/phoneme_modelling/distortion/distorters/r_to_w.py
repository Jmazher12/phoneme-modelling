from pathlib import Path
from typing import Optional

import parselmouth
import librosa
import numpy as np

from phoneme_modelling.distortion.distorter import AudioDistorter


class RToWDistorter(AudioDistorter):

    @staticmethod
    def get_name() -> str:
        return "r_to_w"

    def is_restricted_to_phonemes(self) -> Optional[list[str]]:
        """Returns list of phoneme texts which this distortion can be used for, returns null if this is valid for any audio"""
        return ["r"]

    def distort(
        self, 
        original_file_path: Path,
        original_phoneme_text: str,
        original_phoneme_ipa: str,
        distortion_intensity: float,
        distorted_file_path: Path,
    ) -> tuple[bool, Optional[str]]:
        """
        Applies distortion to original audio with desired intensity and saves distorted audio to new file, returning 
        true if successful or false if the audio was rejected and optionally the ipa representation for the distorted
        audio.
        """
        slider: Slider
        clip: Clip
        index: int
        level: int
        donor: Clip | None
        donor_index: int | None
        rng: np.random.Generator

        alpha = alpha_for(level)
        has_donor = donor is not None and donor_index is not None
        label = slider.produced_ipa

        targets = default_formant_targets("w", clip)
        if has_donor:
            targets |= measure_formants(donor, donor_index)
        params = {"alpha": alpha, "F2_target_hz": round(targets[2]), "F3_target_hz": round(targets[3])}
        params["target_source"] = "donor" if has_donor else "default"
        distorted = shift_formants(clip, index, {2: targets[2], 3: targets[3]}, alpha=alpha, label=label), params
        return True, None


HIGH_VOICE_F0 = 165.0
MIN_FORMANT_HZ = 100.0
DEFAULT_FORMANT_TARGETS: dict[str, dict[str, tuple[float, float]]] = {
    "w": {"low": (750.0, 2350.0), "high": (900.0, 2700.0)},
    "l": {"low": (1000.0, 2700.0), "high": (1150.0, 3000.0)},
    "j": {"low": (2100.0, 2900.0), "high": (2500.0, 3200.0)},
}

def to_samples(sample_rate: int, milliseconds: float) -> int:
    return int(round(sample_rate * milliseconds / 1000))

def median_f0(samples: np.ndarray, sample_rate: int) -> float | None:
    if len(samples) < to_samples(sample_rate, 40):
        return None
    pitch = parselmouth.Sound(samples, sample_rate).to_pitch(pitch_floor=75.0, pitch_ceiling=600.0)
    voiced = pitch.selected_array["frequency"]
    voiced = voiced[voiced > 0]
    return float(np.median(voiced)) if len(voiced) else None

def default_formant_targets(phone: str, clip: Clip) -> dict[int, float]:
    f0 = median_f0(clip.samples, clip.sample_rate)
    voice = "high" if f0 is not None and f0 > HIGH_VOICE_F0 else "low"
    second, third = DEFAULT_FORMANT_TARGETS[phone][voice]
    return {2: second, 3: third}

def max_formant_for(clip: Clip) -> float:
    """Praat's usual formant ceiling: 5000 Hz for low voices, 5500 Hz for higher voices."""
    f0 = median_f0(clip.samples, clip.sample_rate)
    return 5500.0 if f0 is not None and f0 > HIGH_VOICE_F0 else 5000.0

def measure_formants(clip: Clip, index: int, formants: tuple[int, ...] = (2, 3)) -> dict[int, float]:
    """Median formant values over the middle half of segment ``index``."""
    segment = clip.segments[index]
    sound = parselmouth.Sound(clip.samples, clip.sample_rate)
    formant = sound.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=max_formant_for(clip))
    quarter = (segment.end - segment.start) / 4
    times = np.arange(segment.start + quarter, segment.end - quarter, 0.005)
    measured = {}
    for number in formants:
        values = np.array([formant.get_value_at_time(number, time) for time in times])
        values = values[np.isfinite(values)]
        if len(values):
            measured[number] = float(np.median(values))
    return measured

def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) >= length:
        return samples[:length]
    return np.pad(samples, (0, length - len(samples)))


def shift_formants(
    clip: Clip,
    index: int,
    targets: dict[int, float],
    alpha: float,
    ramp_ms: float = 40.0,
    label: str | None = None,
) -> Clip:
    """Pull formant tracks of segment ``index`` a fraction ``alpha`` of the way toward ``targets``.
    The weight is 1 inside the segment and ramps to 0 over ``ramp_ms`` into its
    neighbors, so transitions into the vowel move with the consonant. Only the
    band below the formant ceiling is resynthesized; higher frequencies are kept
    from the original so the clip does not sound muffled.
    """
    rate = clip.sample_rate
    segment = clip.segments[index]
    ramp = ramp_ms / 1000
    max_formant = max_formant_for(clip)
    analysis_rate = 2 * max_formant

    sound = parselmouth.Sound(clip.samples, rate)
    resampled = parselmouth.praat.call(sound, "Resample", analysis_rate, 50)
    lpc = parselmouth.praat.call(resampled, "To LPC (burg)", 12, 0.025, 0.005, 50)
    source = parselmouth.praat.call([resampled, lpc], "Filter (inverse)")
    formant = parselmouth.praat.call(resampled, "To Formant (burg)", 0.005, 5, max_formant, 0.025, 50)
    grid = parselmouth.praat.call(formant, "Down to FormantGrid")

    region_start = max(0.0, segment.start - ramp)
    region_end = min(clip.duration, segment.end + ramp)
    times = np.arange(region_start, region_end, 0.005)
    for number, target in targets.items():
        values = [parselmouth.praat.call(formant, "Get value at time", number, time, "hertz", "linear") for time in times]
        parselmouth.praat.call(grid, "Remove formant points between", number, region_start, region_end)
        for time, value in zip(times, values, strict=True):
            if not np.isfinite(value):
                continue
            weight = min(1.0, max(0.0, (time - (segment.start - ramp)) / ramp))
            weight *= min(1.0, max(0.0, ((segment.end + ramp) - time) / ramp))
            shifted = max(MIN_FORMANT_HZ, value + alpha * weight * (target - value))  # alpha > 1 can overshoot
            parselmouth.praat.call(grid, "Add formant point", number, time, shifted)

    filtered = parselmouth.praat.call([source, grid], "Filter").values[0]
    filtered = _fit_length(librosa.resample(filtered, orig_sr=analysis_rate, target_sr=rate), len(clip.samples))

    pad = to_samples(rate, 10)
    start = max(0, int(region_start * rate) - pad)
    end = min(len(clip.samples), int(region_end * rate) + pad)
    cutoff = 0.9 * max_formant
    original = clip.samples[start:end]
    new_low = lowpass(filtered[start:end], rate, cutoff)
    new_low *= rms(lowpass(original, rate, cutoff)) / (rms(new_low) + 1e-12)
    region = new_low + highpass(original, rate, cutoff)
    return replace_span(clip, start, end, region, fade_ms=10.0, relabel={index: label} if label else None)