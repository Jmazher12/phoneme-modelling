import numpy as np
import parselmouth

from phoneme_modelling.util.acoustics import dsp
from phoneme_modelling.util.acoustics.clip import VOWEL_LABEL, Clip, Segment

MEASURE_BAND = (500.0, 7500.0)
SHAPE_GRID = np.arange(MEASURE_BAND[0], MEASURE_BAND[1] + 1, 50.0)
SHAPE_SMOOTH_HZ = 300.0
ACTIVE_DB = -35.0
MERGE_GAP_MS = 80.0
MIN_VOWEL_MS = 30.0  # shorter voiced stretches after a release are not treated as a vowel
FFT_SIZE = 2048
BURST_FLOOR_DB = 50.0
BURST_RANGE_DB = 30.0
BURST_RISE_SHARE = 0.5
FRICATION_FLOOR_DB = 10.0

NAN = float("nan")


def frame_db(samples: np.ndarray, rate: int, frame_ms: float = 10.0) -> np.ndarray:
    """Frame RMS in dB relative to the clip's peak."""
    hop = dsp.to_samples(rate, frame_ms)
    peak = float(np.max(np.abs(samples), initial=0.0)) + 1e-12
    frames = max(1, len(samples) // hop)
    values = np.array([dsp.rms(samples[i * hop : (i + 1) * hop]) for i in range(frames)])
    return 20 * np.log10(values / peak + 1e-12)


def active_span(samples: np.ndarray, rate: int) -> tuple[int, int]:
    """Sample range of the main sound: frames above ACTIVE_DB, gaps < MERGE_GAP_MS merged, most energetic run kept."""
    hop = dsp.to_samples(rate, 10.0)
    active = frame_db(samples, rate) > ACTIVE_DB
    runs: list[list[int]] = []
    for index in np.flatnonzero(active):
        if runs and index - runs[-1][1] <= MERGE_GAP_MS / 10:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    if not runs:
        return 0, len(samples)
    energy = [np.sum(samples[a * hop : (b + 1) * hop] ** 2) for a, b in runs]
    first, last = runs[int(np.argmax(energy))]
    return max(0, (first - 1) * hop), min(len(samples), (last + 2) * hop)


def voicing(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Praat autocorrelation voicing decision every 5 ms: (times in s, voiced flags)."""
    if len(samples) < dsp.to_samples(rate, 60):
        return np.zeros(0), np.zeros(0, dtype=bool)
    pitch = parselmouth.Sound(samples, rate).to_pitch_ac(time_step=0.005, pitch_floor=75.0, pitch_ceiling=600.0)
    return np.asarray(pitch.xs()), pitch.selected_array["frequency"] > 0


def first_voiced_run(times: np.ndarray, voiced: np.ndarray, after: float, min_ms: float) -> tuple[float, float] | None:
    """First run of voiced frames starting at or after ``after`` that lasts at least ``min_ms``."""
    start = None
    for time, flag in zip(times, voiced, strict=True):
        if time < after:
            continue
        if flag and start is None:
            start = time
        elif not flag and start is not None:
            if time - start >= min_ms / 1000:
                return start, time
            start = None
    if start is not None and times[-1] - start >= min_ms / 1000:
        return start, float(times[-1])
    return None


def burst_time(samples: np.ndarray, rate: int, start: int, end: int) -> float:
    """Stop release: the first rise in > 2 kHz energy (1 ms frames, 3 ms smoothing) that is at least
    BURST_RISE_SHARE of the steepest rise inside [start, end)."""
    high = dsp.highpass(samples[start:end], rate, 2000.0, order=4)
    hop = dsp.to_samples(rate, 1.0)
    frames = len(high) // hop
    if frames < 5:
        return start / rate
    energy = 10 * np.log10(np.array([np.sum(high[i * hop : (i + 1) * hop] ** 2) for i in range(frames)]) + 1e-20)
    # Floor the track so a jump out of digital silence (common in TTS output) is not taken for the burst,
    # and only accept rises that reach a frame within BURST_RANGE_DB of the loudest one.
    energy = np.maximum(energy, energy.max() - BURST_FLOOR_DB)
    energy = np.convolve(np.pad(energy, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    rise = np.diff(energy)
    rise[energy[1:] < energy.max() - BURST_RANGE_DB] = -np.inf
    # The release is the first abrupt onset; later rises (e.g. loud aspiration) must not win over it.
    first = int(np.flatnonzero(rise >= BURST_RISE_SHARE * rise.max())[0])
    return (start + (first + 1) * hop) / rate


def frication_span(samples: np.ndarray, rate: int) -> tuple[float, float]:
    """Longest run of 5 ms frames whose 2.5-7.5 kHz energy exceeds their 0.3-1.5 kHz energy (hiss, not voice).

    Frames must also be FRICATION_FLOOR_DB above the clip's noise floor (10th percentile frame level), which keeps
    weak /θ/ that the activity threshold drops. Gaps up to 20 ms are bridged.
    """
    hop = dsp.to_samples(rate, 5.0)
    window = np.hanning(2 * hop)
    freqs = np.fft.rfftfreq(2 * hop, 1 / rate)
    high_band = (freqs >= 2500) & (freqs <= MEASURE_BAND[1])
    low_band = (freqs >= 300) & (freqs <= 1500)
    frames = max(1, (len(samples) - 2 * hop) // hop)
    ratio, level = np.zeros(frames), np.zeros(frames)
    for i in range(frames):
        power = np.abs(np.fft.rfft(samples[i * hop : i * hop + 2 * hop] * window)) ** 2
        ratio[i] = 10 * np.log10((power[high_band].sum() + 1e-20) / (power[low_band].sum() + 1e-20))
        level[i] = 10 * np.log10(power.sum() + 1e-20)
    floor = np.percentile(level, 10)
    hiss = (ratio > 0.0) & (level > floor + FRICATION_FLOOR_DB) & (level > level.max() - 50.0)
    runs: list[list[int]] = []
    for index in np.flatnonzero(hiss):
        if runs and index - runs[-1][1] <= 4:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    if not runs:
        start, end = active_span(samples, rate)
        return start / rate, end / rate
    first, last = max(runs, key=lambda run: np.sum(10 ** (level[run[0] : run[1] + 1] / 10)))
    return first * hop / rate, min(len(samples), (last + 2) * hop) / rate


def formant_tracks(clip: Clip, start: float, end: float) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """F1-F3 every 5 ms between ``start`` and ``end`` (Burg, ceiling chosen from F0 like the formant shift)."""
    sound = parselmouth.Sound(clip.samples, clip.sample_rate)
    formant = sound.to_formant_burg(
        time_step=0.005, max_number_of_formants=5, maximum_formant=dsp.max_formant_for(clip)
    )
    times = np.arange(start, end, 0.005)
    tracks = {n: np.array([formant.get_value_at_time(n, t) for t in times]) for n in (1, 2, 3)}
    return times, tracks


def segment_fricative(samples: np.ndarray, rate: int, phone: str) -> list[Segment]:
    """The hiss, plus a following vowel when there is one."""
    end_time = active_span(samples, rate)[1] / rate
    times, voiced = voicing(samples, rate)
    hiss_start, hiss_end = frication_span(samples, rate)
    vowel = first_voiced_run(times, voiced, hiss_end - 0.01, 40.0)
    if vowel and vowel[0] - hiss_end < 0.05:
        vowel_end = min(vowel[1], end_time) if vowel[1] > hiss_end else end_time
        if vowel_end - max(hiss_end, vowel[0]) >= MIN_VOWEL_MS / 1000:
            return [Segment(phone, hiss_start, hiss_end), Segment(VOWEL_LABEL, hiss_end, vowel_end)]
    return [Segment(phone, hiss_start, hiss_end)]


def segment_stop(samples: np.ndarray, rate: int, phone: str) -> list[Segment]:
    """Closure and release, plus a following vowel when there is one."""
    start, end = active_span(samples, rate)
    start_time, end_time = start / rate, end / rate
    times, voiced = voicing(samples, rate)
    search_end = min(end, start + dsp.to_samples(rate, 400.0))
    burst = burst_time(samples, rate, start, search_end)
    vowel = first_voiced_run(times, voiced, burst + 0.002, MIN_VOWEL_MS)
    if vowel and vowel[0] - burst < 0.2 and end_time - vowel[0] >= MIN_VOWEL_MS / 1000:
        return [Segment(phone, start_time, vowel[0]), Segment(VOWEL_LABEL, vowel[0], end_time)]
    return [Segment(phone, start_time, end_time)]


def power_spectrum(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Average Hann-windowed power spectrum (FFT_SIZE points, 75% overlap)."""
    window = np.hanning(FFT_SIZE)
    hop = FFT_SIZE // 4
    if len(samples) < FFT_SIZE:
        samples = np.pad(samples, (0, FFT_SIZE - len(samples)))
    frames = [samples[i : i + FFT_SIZE] * window for i in range(0, len(samples) - FFT_SIZE + 1, hop)]
    power = np.mean([np.abs(np.fft.rfft(frame)) ** 2 for frame in frames], axis=0)
    return np.fft.rfftfreq(FFT_SIZE, 1 / rate), power


def spectral_moments(freqs: np.ndarray, power: np.ndarray) -> dict[str, float]:
    """Centre of gravity, SD, skewness and excess kurtosis of the power spectrum inside MEASURE_BAND."""
    band = (freqs >= MEASURE_BAND[0]) & (freqs <= MEASURE_BAND[1])
    f, p = freqs[band], power[band]
    p = p / (p.sum() + 1e-30)
    cog = float((f * p).sum())
    sd = float(np.sqrt(((f - cog) ** 2 * p).sum()))
    skew = float(((f - cog) ** 3 * p).sum() / (sd**3 + 1e-12))
    kurtosis = float(((f - cog) ** 4 * p).sum() / (sd**4 + 1e-12) - 3.0)
    return {"cog_hz": cog, "sd_hz": sd, "skewness": skew, "kurtosis": kurtosis}


def smoothed_db(freqs: np.ndarray, power: np.ndarray, grid: np.ndarray = SHAPE_GRID) -> np.ndarray:
    """Power spectrum smoothed over SHAPE_SMOOTH_HZ, in dB, sampled on ``grid``."""
    width = max(1, int(round(SHAPE_SMOOTH_HZ / (freqs[1] - freqs[0]))))
    smooth = np.convolve(power, np.ones(width) / width, mode="same")
    return np.interp(grid, freqs, 10 * np.log10(smooth + 1e-20))


def stop_measures(clip: Clip, index: int) -> dict[str, object]:
    """Burst, closure voicing, VOT (or open-release noise length), release voicing and burst spectrum."""
    rate = clip.sample_rate
    start, end = clip.bounds(index)
    burst = burst_time(clip.samples, rate, start, min(end, start + dsp.to_samples(rate, 400.0)))
    times, voiced = voicing(clip.samples, rate)
    seg = clip.segments[index]
    before = (times >= seg.start) & (times < burst)
    closure_voicing = 5.0 * float(np.sum(voiced[before]))

    open_release = not clip.is_vowel(index + 1)
    if open_release:
        # No vowel: VOT is undefined, so measure how long the release noise lasts (> 1 kHz, within 20 dB of its max).
        high = dsp.highpass(clip.samples[int(burst * rate) : end], rate, 1000.0, order=4)
        db = frame_db(high, rate, 2.0) if len(high) else np.zeros(1)
        loud = np.flatnonzero(db > -20.0)
        vot = 2.0 * (loud[-1] + 1) if len(loud) else 0.0
    else:
        vot = 1000 * (clip.segments[index + 1].start - burst)
    window = (times >= burst + 0.005) & (times < burst + 0.025)
    release_voicing = float(np.mean(voiced[window])) if window.any() else NAN

    burst_samples = clip.samples[int(burst * rate) : int((burst + 0.01) * rate)]
    burst_cog = spectral_moments(*power_spectrum(burst_samples, rate))["cog_hz"] if len(burst_samples) else NAN
    return {
        "closure_voicing_ms": closure_voicing,
        "vot_ms": float(vot),
        "release_voicing": release_voicing,
        "burst_cog_hz": burst_cog,
    }
