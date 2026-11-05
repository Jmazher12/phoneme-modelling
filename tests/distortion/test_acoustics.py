import numpy as np
import pytest
from conftest import RATE, hiss, padded, seconds, stop, voiced_sound

from phoneme_modelling.phonemes import VALID_SOUND_TEXTS
from phoneme_modelling.util.acoustics import landmarks
from phoneme_modelling.util.acoustics.clip import Clip, Segment
from phoneme_modelling.util.acoustics.delivery import as_clip, deliver, stable_rng
from phoneme_modelling.util.acoustics.measures import SOUNDS, measure_as, measure_sound, segment_sound
from phoneme_modelling.util.acoustics.retime import guard, projection, retime, shift_pitch


def test_every_repo_sound_has_an_acoustic_description() -> None:
    assert set(SOUNDS) == {text for text in VALID_SOUND_TEXTS if text.islower()}


def test_active_span_finds_the_sound_inside_silence(rng: np.random.Generator) -> None:
    start, end = landmarks.active_span(padded(voiced_sound(0.5), rng), RATE)
    assert abs(start - seconds(0.4)) < seconds(0.03)
    assert abs(end - seconds(0.9)) < seconds(0.03)


def test_segment_sound_splits_a_stop_from_its_vowel(rng: np.random.Generator) -> None:
    clip = segment_sound(stop(rng), RATE, "p")
    assert len(clip.segments) == 2
    assert clip.is_vowel(1)


def test_measure_sound_gives_every_measure_and_nan_when_it_does_not_apply(rng: np.random.Generator) -> None:
    values = measure_as(padded(hiss(0.5, 4000.0, rng), rng), RATE, "s")
    assert np.isfinite(values["cog_hz"]) and values["cog_hz"] > 3000
    assert np.isnan(values["vot_ms"])


def test_measure_sound_reads_voice_pitch(rng: np.random.Generator) -> None:
    clip = segment_sound(padded(voiced_sound(0.5, f0=140.0), rng), RATE, "m")
    assert measure_sound(clip, "m")["f0_hz"] == pytest.approx(140.0, abs=5.0)


def test_deliver_makes_a_standard_clip_with_the_same_speech_level(rng: np.random.Generator) -> None:
    samples = padded(voiced_sound(0.5), rng)
    delivered = deliver(samples, as_clip(samples), stable_rng("clip"))
    assert len(delivered) == seconds(2.5)
    assert np.max(np.abs(delivered)) == pytest.approx(np.max(np.abs(samples)), rel=0.05)


def test_deliver_lengthens_long_clips_instead_of_cutting_them(rng: np.random.Generator) -> None:
    samples = padded(voiced_sound(3.0), rng)
    assert len(deliver(samples, as_clip(samples), stable_rng("clip"))) > seconds(3.0)


def test_stable_rng_repeats_for_the_same_name() -> None:
    assert stable_rng("a", 1).random() == stable_rng("a", 1).random()
    assert stable_rng("a", 1).random() != stable_rng("a", 2).random()


@pytest.mark.parametrize("key, factor", [("m", 1.5), ("m", 0.6), ("s", 1.5), ("p", 1.5)])
def test_retime_reaches_the_requested_length(key: str, factor: float, rng: np.random.Generator) -> None:
    sounds = {"m": padded(voiced_sound(0.5), rng), "s": padded(hiss(0.6, 4000.0, rng), rng), "p": stop(rng)}
    samples = sounds[key]
    start, end = landmarks.active_span(samples, RATE)
    stretched = retime(samples, RATE, key, factor, rng)
    new_start, new_end = landmarks.active_span(stretched, RATE)
    assert (new_end - new_start) / (end - start) == pytest.approx(factor, rel=0.2)


def test_shift_pitch_moves_f0_without_changing_length(rng: np.random.Generator) -> None:
    samples = padded(voiced_sound(0.5, f0=140.0), rng)
    shifted = shift_pitch(samples, RATE, 2.0)
    assert len(shifted) == len(samples)
    assert measure_as(shifted, RATE, "m")["f0_hz"] == pytest.approx(140.0 * 2 ** (2 / 12), abs=6.0)


def test_projection_is_a_share_of_the_natural_gap() -> None:
    assert projection(100.0, 150.0, 100.0, ratio=False) == pytest.approx(0.5)
    assert projection(100.0, 100.0 * np.e, 2.0, ratio=True) == pytest.approx(0.5)
    assert np.isnan(projection(float("nan"), 1.0, 1.0, ratio=False))


def test_guard_limits_movement_toward_the_other_label() -> None:
    checks = [("cog_hz_diff", 100.0, False)]
    # An error clip must stay an error: moving back toward the correct sound by more than 25% of the gap fails.
    assert guard({"cog_hz_diff": 100.0}, {"cog_hz_diff": 80.0}, checks, is_error=True)
    assert not guard({"cog_hz_diff": 100.0}, {"cog_hz_diff": 70.0}, checks, is_error=True)
    # A correct clip must stay correct: moving toward the error by more than 25% of the gap fails.
    assert guard({"cog_hz_diff": 100.0}, {"cog_hz_diff": 120.0}, checks, is_error=False)
    assert not guard({"cog_hz_diff": 100.0}, {"cog_hz_diff": 130.0}, checks, is_error=False)


def test_clip_bounds_stay_inside_the_recording() -> None:
    clip = Clip(np.zeros(100), 100, [Segment("x", 0.5, 2.0)])
    assert clip.bounds(0) == (50, 100)
