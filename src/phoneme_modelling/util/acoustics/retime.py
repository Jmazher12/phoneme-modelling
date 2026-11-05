import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from phoneme_modelling.util.acoustics import dsp, landmarks
from phoneme_modelling.util.acoustics.measures import SOUNDS, burst_window, segment_sound

PROTECT_BURST_BEFORE_MS = 3.0
PROTECT_BURST_AFTER_MS = 15.0
PROTECT_OFFSET_MS = 20.0
MAX_ONSET_SHARE = 0.3
# 60 ms keeps a fricative's onset slower than an affricate's (20-40 ms) while still letting long crescendos shrink.
MAX_PROTECTED_ONSET_MS = 60.0
MIN_STRETCH_MS = 25.0
REGION_FACTOR_LIMITS = (0.1, 4.0)
VOICED_REGION_SHARE = 0.5
GUARD_TOWARD = 0.25
GUARD_AWAY = 1.0
MAX_SEMITONES = 4.0


def _stretch_noise(x: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    """Random-phase resynthesis of the interpolated magnitude: noise stays noise (overlap-add of noise sounds buzzy)."""
    n_fft = 512 if len(x) >= 1024 else 256
    if len(x) < n_fft:
        return x.copy()
    hop = n_fft // 4
    magnitude = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    frames = magnitude.shape[1]
    positions = np.linspace(0, frames - 1, max(2, int(round(frames * factor))))
    low = np.floor(positions).astype(int)
    high = np.minimum(low + 1, frames - 1)
    weight = positions - low
    stretched = magnitude[:, low] * (1 - weight) + magnitude[:, high] * weight
    phase = np.exp(2j * np.pi * rng.random(stretched.shape))
    y = librosa.istft(stretched * phase, hop_length=hop, length=int(round(len(x) * factor)))
    return y * (dsp.rms(x) / (dsp.rms(y) + 1e-12))


def _stretch_voiced(x: np.ndarray, rate: int, factor: float) -> np.ndarray:
    """Praat overlap-add, which keeps the pitch."""
    return call(parselmouth.Sound(x, rate), "Lengthen (overlap-add)", 75, 600, factor).values[0]


def _voiced_share(x: np.ndarray, rate: int) -> float:
    _, voiced = landmarks.voicing(x, rate)
    return float(np.mean(voiced)) if len(voiced) else 0.0


def _stretch(x: np.ndarray, rate: int, factor: float, rng: np.random.Generator) -> np.ndarray:
    if abs(factor - 1.0) < 0.02:
        return x.copy()
    if _voiced_share(x, rate) >= VOICED_REGION_SHARE and len(x) >= dsp.to_samples(rate, 60):
        return _stretch_voiced(x, rate, factor)
    return _stretch_noise(x, factor, rng)


def _onset_ms(x: np.ndarray, rate: int) -> float:
    """Time for the 2 ms envelope to first reach 90% of its maximum, plus 10 ms."""
    hop = dsp.to_samples(rate, 2.0)
    frames = len(x) // hop
    if frames < 3:
        return 0.0
    envelope = np.convolve([dsp.rms(x[i * hop : (i + 1) * hop]) for i in range(frames)], np.ones(3) / 3, mode="same")
    return 2.0 * int(np.argmax(envelope >= 0.9 * envelope.max())) + 10.0


def _regions(samples: np.ndarray, rate: int, key: str) -> tuple[int, int, list[tuple[int, int]]]:
    """(speech start, speech end, stretchable regions) for sound ``key``.

    Continuants stretch in their steady middle (the onset rise and last 20 ms are kept); stops and affricates stretch
    in their prevoicing and post-release part (the burst is kept).
    """
    start, end = landmarks.active_span(samples, rate)
    manner = SOUNDS[key].manner
    if manner in ("stop", "affricate"):
        _, _, burst_s = burst_window(segment_sound(samples, rate, key))
        burst = int(burst_s * rate)
        before = burst - dsp.to_samples(rate, PROTECT_BURST_BEFORE_MS)
        after = burst + dsp.to_samples(rate, PROTECT_BURST_AFTER_MS)
        regions = [(start, before), (after, end)]
    else:
        onset = dsp.to_samples(rate, min(_onset_ms(samples[start:end], rate), MAX_PROTECTED_ONSET_MS))
        onset = min(onset, int(MAX_ONSET_SHARE * (end - start)))
        regions = [(start + onset, end - dsp.to_samples(rate, PROTECT_OFFSET_MS))]
    minimum = dsp.to_samples(rate, MIN_STRETCH_MS)
    return start, end, [(s, e) for s, e in regions if e - s >= minimum]


def retime(samples: np.ndarray, rate: int, key: str, factor: float, rng: np.random.Generator) -> np.ndarray:
    """Change the speech duration by ``factor`` using only the stretchable regions of sound ``key``."""
    start, end, regions = _regions(samples, rate, key)
    if not regions or abs(factor - 1.0) < 0.02:
        return samples.copy()
    total = end - start
    stretchable = sum(e - s for s, e in regions)
    wanted = factor * total
    region_factor = float(np.clip((wanted - (total - stretchable)) / stretchable, *REGION_FACTOR_LIMITS))
    pieces, cursor = [], 0
    for s, e in sorted(regions):
        pieces.append(samples[cursor:s])
        stretched = _stretch(samples[s:e], rate, region_factor, rng)
        fade = min(dsp.to_samples(rate, 5.0), len(stretched) // 4)
        if fade and len(pieces[-1]) >= fade:
            # Short crossfades into and out of the stretched region avoid clicks.
            ramp = dsp.ramp(fade)
            stretched[:fade] = stretched[:fade] * ramp + samples[s : s + fade] * (1 - ramp)
            stretched[-fade:] = stretched[-fade:] * ramp[::-1] + samples[e - fade : e] * (1 - ramp[::-1])
        pieces.append(stretched)
        cursor = e
    pieces.append(samples[cursor:])
    return np.concatenate(pieces)


def shift_pitch(samples: np.ndarray, rate: int, semitones: float) -> np.ndarray:
    """PSOLA over the whole clip. Only for voiced sounds: PSOLA can add voicing to voiceless frication."""
    if abs(semitones) < 0.05:
        return samples.copy()
    sound = parselmouth.Sound(samples, rate)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    tier = call(manipulation, "Extract pitch tier")
    call(tier, "Multiply frequencies", sound.xmin, sound.xmax, 2 ** (semitones / 12))
    call([tier, manipulation], "Replace pitch tier")
    shifted = call(manipulation, "Get resynthesis (overlap-add)").values[0]
    return dsp.fit_length(shifted, len(samples))


def projection(before: float, after: float, natural: float, ratio: bool) -> float:
    """Movement of one cue as a share of the natural gap (positive = toward the produced sound)."""
    if not (np.isfinite(before) and np.isfinite(after) and np.isfinite(natural)) or natural == 0:
        return float("nan")
    if ratio:
        if before <= 0 or after <= 0:
            return float("nan")
        return float(np.log(after / before) / natural)
    return float((after - before) / natural)


def guard(
    before: dict[str, object], after: dict[str, object], checks: list[tuple[str, float, bool]], is_error: bool
) -> bool:
    """Whether a nuisance edit left the defining cues alone. ``checks``: (cue, natural gap, compared as ratio).

    An edit may move a cue at most GUARD_TOWARD of the natural gap toward the other label and GUARD_AWAY away from it.
    """
    passed = True
    for cue, natural, ratio in checks:
        share = projection(float(before.get(cue, np.nan)), float(after.get(cue, np.nan)), natural, ratio)
        if not np.isfinite(share):
            continue
        low, high = (-GUARD_TOWARD, GUARD_AWAY) if is_error else (-GUARD_AWAY, GUARD_TOWARD)
        passed &= low <= share <= high
    return passed
