"""The edits the pair distorters are built from, one per kind of sound change.

Each recipe takes a segmented clip of the source sound and ``alpha``, the share of the natural difference between the
two sounds to apply, and returns the edited samples. Every amount is ``alpha`` x a value from the pair's settings file,
except constants marked fixed. Choices that differ between pairs (which formants move, whether voicing is added or
removed, whether a stop also changes place) are arguments the pair's distorter passes explicitly.
"""

from typing import Literal, Optional

import librosa
import numpy as np

from phoneme_modelling.schema.pair_parameters import PairParameters
from phoneme_modelling.util.acoustics import dsp, landmarks
from phoneme_modelling.util.acoustics.clip import Clip, Segment
from phoneme_modelling.util.acoustics.measures import VOICEBAR_BAND, burst_window

Voicing = Optional[Literal["on", "off", "stronger"]]

GAIN_LIMIT_DB = 30.0
VOICE_SOURCE_CUTOFF_HZ = 600.0  # fixed: low harmonics carry the voice bar
VOICING_AM_DEPTH = 0.5  # fixed: full-strength F0 amplitude modulation of voiced frication
BURST_MS = 10.0
FADE_MS = 5.0
MIN_RELEASE_MS = 8.0


# ---------------------------------------------------------------------------
# Primitives (arrays in, arrays out)
# ---------------------------------------------------------------------------


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


def _apply_shape(
    x: np.ndarray, rate: int, alpha: float, parameters: PairParameters, key: str = "span_shape_db"
) -> np.ndarray:
    """Move the spectrum of ``x`` alpha of the way along the natural shape difference ``key`` (unchanged if missing)."""
    shape = parameters.shape(key)
    if shape is None:
        return x
    return _stft_gain(x, rate, np.asarray(parameters.grid_hz), alpha * shape)


def _splice(
    samples: np.ndarray, start: int, end: int, new: np.ndarray, rate: int, fade_ms: float = FADE_MS
) -> np.ndarray:
    """Replace samples[start:end] by ``new`` with short crossfades at both joints."""
    clip = Clip(samples, rate, [Segment("x", start / rate, end / rate)])
    return dsp.replace_span(clip, start, end, new, fade_ms=fade_ms).samples


def _burst_start(clip: Clip) -> int:
    start, end = clip.speech_bounds()
    first, _, _ = burst_window(
        Clip(clip.samples, clip.sample_rate, [Segment("x", start / clip.sample_rate, end / clip.sample_rate)])
    )
    return int(np.clip(first, start, end - 1))


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
    x: np.ndarray,
    rate: int,
    alpha: float,
    target_vb: float,
    f0: float,
    rng: np.random.Generator,
    am_depth: float = VOICING_AM_DEPTH,
) -> np.ndarray:
    """Voice a sound: F0 amplitude modulation of depth alpha x ``am_depth``, plus a voice bar that moves the
    < 400 Hz / 0.5-7.5 kHz ratio alpha of the way to ``target_vb``."""
    before = _voicebar_db(x, rate)
    wanted = before + alpha * (target_vb - before)
    phase = 2 * np.pi * f0 * np.arange(len(x)) / rate
    depth = alpha * am_depth
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


def _change_voicing(
    x: np.ndarray,
    rate: int,
    alpha: float,
    parameters: PairParameters,
    voicing: Voicing,
    voice_f0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Voicing on, off, or a stronger voice bar (no F0 modulation) for a voiced sound that becomes a more voiced one."""
    if voicing == "on":
        return _add_voicing(x, rate, alpha, parameters.target("voicebar_db", 10.0), voice_f0, rng)
    if voicing == "stronger":
        return _add_voicing(x, rate, alpha, parameters.target("voicebar_db", 10.0), voice_f0, rng, am_depth=0.0)
    if voicing == "off":
        return _remove_voicing(x, rate, alpha, parameters.target("voicebar_db", -30.0), rng)
    return x


def _prevoicing(
    length_ms: float, level_db: float, burst_rms: float, f0: float, rate: int, rng: np.random.Generator
) -> np.ndarray:
    length = dsp.to_samples(rate, max(0.0, length_ms))
    bar = _voice_source(length, rate, f0, rng)
    if not length or dsp.rms(bar) == 0:
        return np.zeros(0)
    bar *= burst_rms * 10 ** (level_db / 20) / dsp.rms(bar)
    return bar * dsp.envelope(length, dsp.to_samples(rate, 10), dsp.to_samples(rate, 2))


def _target_noise(parameters: PairParameters, length: int, rate: int, rng: np.random.Generator) -> np.ndarray:
    """White noise with the produced sound's typical burst spectrum (aspiration and bursts are broadband)."""
    if not length:
        return np.zeros(0)
    grid = np.asarray(parameters.grid_hz)
    shape = parameters.shape("target_primary_shape_db")
    shape = np.zeros(len(grid)) if shape is None else shape
    return _stft_gain(rng.standard_normal(length), rate, grid, shape - shape.mean())


def _decay_envelope(length: int, rate: int, tau_ms: float) -> np.ndarray:
    return np.exp(-np.arange(length) / max(1.0, dsp.to_samples(rate, tau_ms)))


def _abrupt_onset(x: np.ndarray, rate: int, alpha: float, target_rise_ms: float) -> np.ndarray:
    """Make the onset rise in ``target_rise_ms`` instead of the clip's own rise (alpha of the way, in dB)."""
    hop = dsp.to_samples(rate, 2.0)
    frames = len(x) // hop
    if frames < 4:
        return x
    envelope = np.array([dsp.rms(x[i * hop : (i + 1) * hop]) for i in range(frames)])
    envelope = np.convolve(envelope, np.ones(3) / 3, mode="same") + 1e-9
    top_index = int(np.argmax(envelope >= 0.9 * envelope.max()))
    if 2.0 * top_index <= target_rise_ms:
        return x
    level = envelope[top_index]
    wanted = np.full(frames, level)
    ramp = max(1, int(target_rise_ms / 2))
    wanted[:ramp] = level * np.linspace(0.3, 1.0, ramp)
    gain_db = np.zeros(frames)
    gain_db[:top_index] = np.clip(
        alpha * 20 * np.log10(wanted[:top_index] / envelope[:top_index]), -GAIN_LIMIT_DB, GAIN_LIMIT_DB
    )
    gain = np.interp(np.arange(len(x)), np.arange(frames) * hop + hop / 2, 10 ** (gain_db / 20))
    return x * gain


# ---------------------------------------------------------------------------
# Recipes (one per kind of sound change)
# ---------------------------------------------------------------------------


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


def reshape_fricative(clip: Clip, alpha: float, parameters: PairParameters) -> np.ndarray:
    """Hiss shape gain only (both sounds voiced alike)."""
    start, end = clip.speech_bounds()
    x = _apply_shape(clip.samples[start:end], clip.sample_rate, alpha, parameters)
    return _splice(clip.samples, start, end, x, clip.sample_rate)


def voice_fricative(
    clip: Clip, alpha: float, parameters: PairParameters, voice_f0: float, rng: np.random.Generator
) -> np.ndarray:
    """Hiss shape gain, then voicing on."""
    start, end = clip.speech_bounds()
    rate = clip.sample_rate
    x = _apply_shape(clip.samples[start:end], rate, alpha, parameters)
    x = _change_voicing(x, rate, alpha, parameters, "on", voice_f0, rng)
    return _splice(clip.samples, start, end, x, rate)


def devoice_fricative(clip: Clip, alpha: float, parameters: PairParameters, rng: np.random.Generator) -> np.ndarray:
    """Voicing off, then hiss shape gain."""
    start, end = clip.speech_bounds()
    rate = clip.sample_rate
    x = _remove_voicing(clip.samples[start:end], rate, alpha, parameters.target("voicebar_db", -30.0), rng)
    x = _apply_shape(x, rate, alpha, parameters)
    return _splice(clip.samples, start, end, x, rate)


def change_stop(
    clip: Clip,
    measures: dict[str, object],
    alpha: float,
    parameters: PairParameters,
    voice_f0: float,
    rng: np.random.Generator,
    voicing: Optional[Literal["on", "off"]],
    place: bool,
) -> np.ndarray:
    """Stop to stop. Voicing off: prevoicing faded by (1 - alpha) and alpha x the natural VOT change of aspiration
    replacing the start of the release. Voicing on: alpha x the target's prevoicing at its closure level, release noise
    cut toward the target's VOT. Place: burst and release shape gain alpha x the natural burst shape difference."""
    samples, rate = clip.samples, clip.sample_rate
    start, end = clip.speech_bounds()
    first = _burst_start(clip)
    burst_end = min(end, first + dsp.to_samples(rate, BURST_MS))
    burst_rms = dsp.rms(samples[first:burst_end]) + 1e-12
    pre, release = samples[start:first], samples[burst_end:end]
    blend = min(alpha, 1.0)

    if voicing == "off":
        aspiration_ms = max(0.0, alpha * parameters.change("vot_ms", 0.0))
        level = parameters.target("aspiration_db", 0.0)
        length = dsp.to_samples(rate, aspiration_ms)
        aspiration = _target_noise(parameters, length, rate, rng)
        aspiration *= burst_rms * 10 ** (level / 20) / (dsp.rms(aspiration) + 1e-12)
        aspiration *= dsp.envelope(length, dsp.to_samples(rate, 3), dsp.to_samples(rate, 15))
        # Cutback: the aspiration replaces the start of the voiced release, the rest fades with (1 - alpha).
        remainder = release[min(len(release), length) :] * (1 - blend)
        pre = pre * (1 - blend)
        release = np.concatenate([aspiration, remainder])
    elif voicing == "on":
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
    if place:
        offset = len(pre)
        new = np.concatenate([new[:offset], _apply_shape(new[offset:], rate, alpha, parameters, "primary_shape_db")])
    return _splice(samples, start, end, new, rate, fade_ms=2.0)


def fricative_to_stop(
    clip: Clip,
    alpha: float,
    parameters: PairParameters,
    voice_f0: float,
    rng: np.random.Generator,
    devoice: bool,
    prevoice: bool,
    shape: Literal["primary", "span"],
) -> np.ndarray:
    """Fricative to stop or affricate: abrupt onset, frication cut alpha of the way to the target's release length and
    decaying like a release, shape gain toward the target's burst (``primary``) or hiss (``span``). ``devoice`` removes
    voicing first (voiced fricative to voiceless stop); ``prevoice`` adds the target's prevoicing (voiced stop)."""
    samples, rate = clip.samples, clip.sample_rate
    start, end = clip.speech_bounds()
    x = samples[start:end].copy()
    if devoice:
        x = _remove_voicing(x, rate, alpha, parameters.target("voicebar_db", -30.0), rng)
    x = _abrupt_onset(x, rate, alpha, parameters.target("rise_ms", 5.0))
    release_ms = parameters.target("vot_ms", 40.0)
    own_ms = 1000 * len(x) / rate
    keep_ms = max(MIN_RELEASE_MS, own_ms - alpha * max(0.0, own_ms - release_ms))
    x = x[: min(len(x), dsp.to_samples(rate, keep_ms))]
    # The release dies away over the kept part (a stop's release decays; a fricative's hiss does not).
    blend = min(alpha, 1.0)
    x *= 1 - blend + blend * _decay_envelope(len(x), rate, max(5.0, keep_ms / 3))
    fade = min(dsp.to_samples(rate, FADE_MS), len(x) // 2)
    if fade:
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    x = _apply_shape(x, rate, alpha, parameters, f"{shape}_shape_db")
    if prevoice:
        burst_rms = dsp.rms(x[: dsp.to_samples(rate, BURST_MS)]) + 1e-12
        prevoice_ms = alpha * parameters.target("closure_voicing_ms", 60.0)
        level = parameters.target("closure_level_db", -6.0)
        x = np.concatenate([_prevoicing(prevoice_ms, level, burst_rms, voice_f0, rate, rng), x])
    return _splice(samples, start, end, x, rate, fade_ms=2.0)


def affricate_to_fricative(
    clip: Clip,
    alpha: float,
    parameters: PairParameters,
    voice_f0: float,
    rng: np.random.Generator,
    voicing: Optional[Literal["off", "stronger"]],
) -> np.ndarray:
    """Affricate to fricative: closure faded by (1 - alpha), burst softened by an onset ramp of alpha x the target's
    rise time, hiss shape gain toward the target's, voicing changed as for fricatives."""
    samples, rate = clip.samples, clip.sample_rate
    start, end = clip.speech_bounds()
    first = _burst_start(clip)
    blend = min(alpha, 1.0)
    pre, x = samples[start:first] * (1 - blend), samples[first:end].copy()
    if voicing == "off":
        x = _change_voicing(x, rate, alpha, parameters, voicing, voice_f0, rng)
    ramp = min(len(x), dsp.to_samples(rate, alpha * parameters.target("rise_ms", 100.0)))
    if ramp > 1:
        x[:ramp] *= 1 - blend * (1 - dsp.ramp(ramp))  # raised-cosine onset at full strength, none at alpha = 0
    x = _apply_shape(x, rate, alpha, parameters)
    if voicing == "stronger":
        x = _change_voicing(x, rate, alpha, parameters, voicing, voice_f0, rng)
    return _splice(samples, start, end, np.concatenate([pre, x]), rate, fade_ms=2.0)


def affricate_to_stop(
    clip: Clip,
    measures: dict[str, object],
    alpha: float,
    parameters: PairParameters,
    voice_f0: float,
    rng: np.random.Generator,
    prevoice: bool,
) -> np.ndarray:
    """Affricate to stop: frication after the burst cut toward the target's release length, burst shape gain toward the
    target's; ``prevoice`` adds the missing share of the target's prevoicing (voiced stop)."""
    samples, rate = clip.samples, clip.sample_rate
    start, end = clip.speech_bounds()
    first = _burst_start(clip)
    x = samples[first:end].copy()
    own_ms = 1000 * len(x) / rate
    release_ms = parameters.target("vot_ms", 60.0)
    keep_ms = max(MIN_RELEASE_MS, own_ms - alpha * max(0.0, own_ms - release_ms))
    x = x[: dsp.to_samples(rate, keep_ms)]
    fade = min(dsp.to_samples(rate, 10), len(x) // 2)
    if fade:
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    x = _apply_shape(x, rate, alpha, parameters, "primary_shape_db")
    pre = samples[start:first]
    if prevoice:
        own_voicing = float(measures.get("closure_voicing_ms", 0.0))
        missing = parameters.target("closure_voicing_ms", 60.0) - (own_voicing if np.isfinite(own_voicing) else 0.0)
        prevoice_ms = alpha * max(0.0, missing)
        level = parameters.target("closure_level_db", -6.0)
        burst_rms = dsp.rms(x[: dsp.to_samples(rate, BURST_MS)]) + 1e-12
        pre = np.concatenate([_prevoicing(prevoice_ms, level, burst_rms, voice_f0, rate, rng), pre])
    return _splice(samples, start, end, np.concatenate([pre, x]), rate, fade_ms=2.0)


def change_nasal_place(clip: Clip, alpha: float, parameters: PairParameters) -> np.ndarray:
    """Murmur shape gain."""
    start, end = clip.speech_bounds()
    x = _apply_shape(clip.samples[start:end], clip.sample_rate, alpha, parameters)
    return _splice(clip.samples, start, end, x, clip.sample_rate)


def nasal_to_stop(
    clip: Clip, alpha: float, parameters: PairParameters, rng: np.random.Generator, voiced: bool
) -> np.ndarray:
    """Nasal to stop: murmur low-passed and shortened toward the target's prevoicing (removed for a voiceless target),
    then a burst with the target's typical spectrum and release length, as loud as the murmur was (x alpha)."""
    samples, rate = clip.samples, clip.sample_rate
    start, end = clip.speech_bounds()
    murmur = samples[start:end]
    murmur_rms = dsp.rms(murmur) + 1e-12
    own_ms = 1000 * len(murmur) / rate
    closure_ms = parameters.target("closure_voicing_ms", 60.0) if voiced else 0.0
    blend = min(alpha, 1.0)
    keep_ms = own_ms - blend * max(0.0, own_ms - closure_ms)
    closure = murmur[: dsp.to_samples(rate, keep_ms)].copy()
    cutoff = float(np.exp((1 - blend) * np.log(4000.0) + blend * np.log(500.0)))
    closure = dsp.lowpass(closure, rate, cutoff, order=4) if len(closure) > 48 else closure
    level = parameters.target("closure_level_db", -6.0) if voiced else -60.0
    closure *= 10 ** (blend * level / 20) * (1.0 if voiced else 1 - blend)
    if len(closure) > 2:
        closure *= dsp.envelope(len(closure), dsp.to_samples(rate, 5), dsp.to_samples(rate, 2))

    release_ms = max(BURST_MS, parameters.target("vot_ms", 20.0))
    length = dsp.to_samples(rate, release_ms)
    noise = _target_noise(parameters, length, rate, rng)
    tau = 3.0 if voiced else release_ms / 3
    noise *= _decay_envelope(length, rate, tau) * dsp.envelope(length, dsp.to_samples(rate, 1), dsp.to_samples(rate, 3))
    noise *= alpha * murmur_rms / (dsp.rms(noise[: dsp.to_samples(rate, BURST_MS)]) + 1e-12)
    return _splice(samples, start, end, np.concatenate([closure, noise]), rate, fade_ms=2.0)
