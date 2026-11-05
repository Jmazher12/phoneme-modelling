"""Audio helpers for the LLM generators, from sara-sound-pipeline (gen_gemini.py, tools.py prep). Needs ffmpeg."""
import subprocess
import wave
from pathlib import Path

import numpy as np


CLIP_SAMPLE_RATE = 16000


def _ffmpeg(args: list[str], data: bytes | None = None) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", *args], input=data, capture_output=True, check=True).stdout


def load_as_pcm16(path: Path, rate: int = 16000) -> bytes:
    """Any audio file as mono 16-bit PCM."""
    return _ffmpeg(["-i", str(path), "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"])


def resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    return _ffmpeg(["-f", "s16le", "-ar", str(from_rate), "-ac", "1", "-i", "-", "-ar", str(to_rate), "-f", "s16le", "-"], pcm)


def save_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm)


def seconds(pcm: bytes, rate: int) -> float:
    return len(pcm) / 2 / rate


def peak(pcm: bytes) -> float:
    """Peak of 16-bit PCM, 0-1."""
    x = np.frombuffer(pcm, dtype=np.int16)
    return float(np.abs(x).max()) / 32768 if len(x) else 0.0


def crop_to_onset(pcm: bytes, rate: int, keep_s: float = 1.8) -> bytes:
    """The first keep_s seconds after the sound starts, with a 150 ms fade-out (models often hold a sound for 30+ s)."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    frame = rate // 100
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-9)
    loud = np.where(rms > rms.max() * 0.01)[0]
    start = max((loud[0] if len(loud) else 0) * frame - int(0.05 * rate), 0)
    y = x[start : start + int(keep_s * rate)].copy()
    fade = min(int(0.15 * rate), len(y))
    y[len(y) - fade :] *= np.linspace(1, 0, fade)
    return y.astype(np.int16).tobytes()


def prep_clip(source: Path, target: Path, seed: int = 0) -> None:
    """tools.py prep: 16 kHz mono, silence trimmed (-40 dB, 80 ms margin), peak 0.8, centred in 2.5 s of -60 dBFS noise."""
    sr = CLIP_SAMPLE_RATE
    audio = np.frombuffer(_ffmpeg(["-i", str(source), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]), dtype=np.float32)

    # trim leading/trailing silence, using 10 ms frame energy relative to the loudest frame
    frame = sr // 100
    n = len(audio) // frame
    if n and np.abs(audio).max() > 0:
        rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
        active = np.where(20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12) > -40.0)[0]
        if len(active):
            margin = int(0.08 * sr)
            audio = audio[max(active[0] * frame - margin, 0) : min((active[-1] + 1) * frame + margin, len(audio))]

    sound = audio / (np.abs(audio).max() or 1.0) * 0.8
    total = max(int(2.5 * sr), len(sound))
    clip = np.random.default_rng(seed).normal(0.0, 10 ** (-60.0 / 20), size=total).astype(np.float32)
    start = (total - len(sound)) // 2
    clip[start : start + len(sound)] += sound
    save_wav(target, (np.clip(clip, -1.0, 1.0) * 32767).astype(np.int16).tobytes(), sr)


def save_raw_and_prepped(pcm: bytes, rate: int, target: Path) -> bool:
    """Saves the model's answer to raw/<name> next to target and the prepped clip to target. Answers over 3 s are cropped
    first (the crop is kept as raw/<stem>_crop.wav). Returns whether it was cropped."""
    raw = target.parent / "raw" / target.name
    save_wav(raw, pcm, rate)
    source, cropped = raw, seconds(pcm, rate) > 3.0
    if cropped:
        source = raw.with_name(f"{target.stem}_crop.wav")
        save_wav(source, crop_to_onset(pcm, rate), rate)
    prep_clip(source, target)
    return cropped
