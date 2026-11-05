import librosa
import numpy as np

from phoneme_modelling.schema.pair_parameters import PairParameters
from phoneme_modelling.util.acoustics import dsp, landmarks
from phoneme_modelling.util.acoustics.clip import Clip, Segment
from phoneme_modelling.util.acoustics.measures import VOICEBAR_BAND, burst_window

GAIN_LIMIT_DB = 30.0
VOICE_SOURCE_CUTOFF_HZ = 600.0  # low harmonics carry the voice bar
VOICING_AM_DEPTH = 0.5  # full-strength F0 amplitude modulation of voiced frication
BURST_MS = 10.0
FADE_MS = 5.0
MIN_RELEASE_MS = 8.0


def _stft_gain(x: np.ndarray, rate: int, grid: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    """Filter ``x`` by a gain curve given on ``grid`` (held flat outside it), every frame alike."""
    n_fft = 1024 if len(x) >= 2048 else 512 if len(x) >= 1024 else 256 if len(x) >= 512 else 128
    if len(x) < n_fft:
        return x.copy()
    hop = n_fft // 4
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
    gain = np.clip(np.interp(freqs, grid, gain_db), -GAIN_LIMIT_DB, GAIN_LIMIT_DB)
    return librosa.istft(stft * (10 ** (gain / 20))[:, None], hop_length=hop, length=len(x))


def _apply_shape(x: np.ndarray, rate: int, alpha: float, parameters: PairParameters) -> np.ndarray:
    """Move the spectrum of ``x`` alpha of the way along the natural difference between the two sounds."""
    return _stft_gain(x, rate, np.asarray(parameters.grid_hz), alpha * np.asarray(parameters.span_shape_db))


def _splice(
    samples: np.ndarray, start: int, end: int, new: np.ndarray, rate: int, fade_ms: float = FADE_MS
) -> np.ndarray:
    """Replace samples[start:end] by ``new`` with short crossfades at both joints."""
    clip = Clip(samples, rate, [Segment("x", start / rate, end / rate)])
    return dsp.replace_span(clip, start, end, new, fade_ms=fade_ms).samples


def _band_power(x: np.ndarray, rate: int, low: float, high: float) -> float:
    freqs, power = landmarks.power_spectrum(x, rate)
    return float(power[(freqs >= low) & (freqs < high)].sum())


def _voicebar_db(x: np.ndarray, rate: int) -> float:
    low = _band_power(x, rate, *VOICEBAR_BAND)
    high = _band_power(x, rate, *landmarks.MEASURE_BAND)
    return float(10 * np.log10((low + 1e-20) / (high + 1e-20)))


def _voice_source(length: int, rate: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    """Low-passed glottal pulse train at ``f0`` with 1% jitter and a slight fall (no two voice bars alike)."""
    if length <= 0:
        return np.zeros(0)
    knots = max(2, length // dsp.to_samples(rate, 20.0) + 2)
    jitter = np.interp(np.arange(length), np.linspace(0, length, knots), rng.standard_normal(knots))
    contour = f0 * np.linspace(1.02, 0.98, length) * (1 + 0.01 * jitter)
    phase = 2 * np.pi * np.cumsum(contour) / rate
    pulses = np.maximum(0.0, np.sin(phase)) ** 3
    pulses -= pulses.mean()
    return dsp.lowpass(pulses, rate, VOICE_SOURCE_CUTOFF_HZ, order=4)


def _noise_resynthesis(x: np.ndarray, rate: int, rng: np.random.Generator, low_gain_db: float = 0.0) -> np.ndarray:
    """Same smoothed spectro-temporal envelope as ``x`` but random phase (a whispered, unvoiced copy).

    The magnitude is smoothed over 400 Hz so harmonics do not survive; ``low_gain_db`` is added below 450 Hz to set
    the voice-bar level.
    """
    n_fft = 512 if len(x) >= 1024 else 256
    if len(x) < n_fft:
        return x.copy()
    hop = n_fft // 4
    magnitude = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    width = max(1, int(round(400.0 / (rate / n_fft))))
    kernel = np.ones(width) / width
    magnitude = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="same"), 0, magnitude)
    magnitude = np.apply_along_axis(lambda row: np.convolve(row, np.ones(3) / 3, mode="same"), 1, magnitude)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
    magnitude[freqs < 450.0] *= 10 ** (low_gain_db / 20)
    phase = np.exp(2j * np.pi * rng.random(magnitude.shape))
    return librosa.istft(magnitude * phase, hop_length=hop, length=len(x))


def _add_voicing(
    x: np.ndarray, rate: int, alpha: float, target_vb: float, f0: float, rng: np.random.Generator
) -> np.ndarray:
    """Voice a voiceless sound: F0 amplitude modulation of depth alpha x 0.5, plus a voice bar that moves the
    < 400 Hz / 0.5-7.5 kHz ratio alpha of the way to ``target_vb``."""
    before = _voicebar_db(x, rate)
    wanted = before + alpha * (target_vb - before)
    phase = 2 * np.pi * f0 * np.arange(len(x)) / rate
    depth = alpha * VOICING_AM_DEPTH
    modulated = x * ((1 - depth / 2) + (depth / 2) * np.cos(phase))
    fade = dsp.to_samples(rate, 15)
    bar = _voice_source(len(x), rate, f0, rng) * dsp.envelope(len(x), fade, fade)
    high = _band_power(modulated, rate, *landmarks.MEASURE_BAND)
    missing = high * 10 ** (wanted / 10) - _band_power(modulated, rate, *VOICEBAR_BAND)
    bar_power = _band_power(bar, rate, *VOICEBAR_BAND)
    gain = np.sqrt(max(0.0, missing) / bar_power) if bar_power > 0 else 0.0
    return modulated + gain * bar


def _remove_voicing(x: np.ndarray, rate: int, alpha: float, target_vb: float, rng: np.random.Generator) -> np.ndarray:
    """Devoice: blend alpha of the way to a noise resynthesis whose low band sits at the target's voice-bar ratio."""
    whisper = _noise_resynthesis(x, rate, rng)
    whisper = _noise_resynthesis(x, rate, rng, low_gain_db=min(0.0, target_vb - _voicebar_db(whisper, rate)))
    whisper *= dsp.rms(x) / (dsp.rms(whisper) + 1e-12)
    blend = min(alpha, 1.0)
    return (1 - blend) * x + blend * whisper


def _prevoicing(
    length_ms: float, level_db: float, burst_rms: float, f0: float, rate: int, rng: np.random.Generator
) -> np.ndarray:
    length = dsp.to_samples(rate, max(0.0, length_ms))
    bar = _voice_source(length, rate, f0, rng)
    if not length or dsp.rms(bar) == 0:
        return np.zeros(0)
    bar *= burst_rms * 10 ** (level_db / 20) / dsp.rms(bar)
    return bar * dsp.envelope(length, dsp.to_samples(rate, 10), dsp.to_samples(rate, 2))


def shift_formants(
    clip: Clip, measures: dict[str, object], alpha: float, parameters: PairParameters, formants: tuple[int, ...]
) -> np.ndarray:
    """Move the listed formants toward (the clip's own value x the natural ratio between the two sounds)."""
    start, end = clip.speech_bounds()
    targets = {}
    for number in formants:
        key = f"f{number}_hz"
        own = float(measures.get(key, np.nan))
        if np.isfinite(own):
            targets[number] = own * float(np.exp(parameters.cue_change[key]))
    if not targets:
        return clip.samples
    rate = clip.sample_rate
    span = Clip(clip.samples, rate, [Segment("x", start / rate, end / rate)])
    return dsp.shift_formants(span, 0, targets, alpha=alpha).samples


def voice_fricative(
    clip: Clip, alpha: float, parameters: PairParameters, voice_f0: float, rng: np.random.Generator
) -> np.ndarray:
    """Hiss shape gain, then voicing on."""
    start, end = clip.speech_bounds()
    rate = clip.sample_rate
    x = _apply_shape(clip.samples[start:end], rate, alpha, parameters)
    x = _add_voicing(x, rate, alpha, parameters.target("voicebar_db", 10.0), voice_f0, rng)
    return _splice(clip.samples, start, end, x, rate)


def devoice_fricative(clip: Clip, alpha: float, parameters: PairParameters, rng: np.random.Generator) -> np.ndarray:
    """Voicing off, then hiss shape gain."""
    start, end = clip.speech_bounds()
    rate = clip.sample_rate
    x = _remove_voicing(clip.samples[start:end], rate, alpha, parameters.target("voicebar_db", -30.0), rng)
    x = _apply_shape(x, rate, alpha, parameters)
    return _splice(clip.samples, start, end, x, rate)


def voice_stop(
    clip: Clip,
    measures: dict[str, object],
    alpha: float,
    parameters: PairParameters,
    voice_f0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add alpha x the target's prevoicing at its closure level, and cut the release noise toward the target's VOT."""
    samples, rate = clip.samples, clip.sample_rate
    start, end = clip.speech_bounds()
    first, _, _ = burst_window(Clip(samples, rate, [Segment("x", start / rate, end / rate)]))
    first = int(np.clip(first, start, end - 1))
    burst_end = min(end, first + dsp.to_samples(rate, BURST_MS))
    burst_rms = dsp.rms(samples[first:burst_end]) + 1e-12
    pre, release = samples[start:first], samples[burst_end:end]

    own_voicing = float(measures.get("closure_voicing_ms", 0.0))
    own_voicing = own_voicing if np.isfinite(own_voicing) else 0.0
    prevoice_ms = alpha * max(0.0, parameters.target("closure_voicing_ms", 60.0) - own_voicing)
    level = parameters.target("closure_level_db", -6.0)
    own_vot = float(measures.get("vot_ms", np.nan))
    target_vot = parameters.target("vot_ms", 15.0)
    keep_ms = own_vot - alpha * (own_vot - target_vot) if np.isfinite(own_vot) else len(release) / rate * 1000
    keep = max(dsp.to_samples(rate, MIN_RELEASE_MS), dsp.to_samples(rate, keep_ms) - (burst_end - first))
    if keep < len(release):
        fade = min(dsp.to_samples(rate, 10), keep)
        release = release[:keep].copy()
        release[-fade:] *= np.linspace(1.0, 0.0, fade)
    pre = np.concatenate([_prevoicing(prevoice_ms, level, burst_rms, voice_f0, rate, rng), pre])
    new = np.concatenate([pre, samples[first:burst_end], release])
    return _splice(samples, start, end, new, rate, fade_ms=2.0)


def change_nasal_place(clip: Clip, alpha: float, parameters: PairParameters) -> np.ndarray:
    """Murmur shape gain."""
    start, end = clip.speech_bounds()
    x = _apply_shape(clip.samples[start:end], clip.sample_rate, alpha, parameters)
    return _splice(clip.samples, start, end, x, clip.sample_rate)
