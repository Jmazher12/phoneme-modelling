from pathlib import Path

import pytest
import soundfile as sf
from conftest import RATE
from test_pair_distorters import run

from phoneme_modelling.distortion import pair_distorter
from phoneme_modelling.distortion.distorters.m_to_n import MToNDistorter, MToNMatchedDistorter
from phoneme_modelling.distortion.distorters.r_to_w import RToWMatchedDistorter
from phoneme_modelling.schema.pair_parameters import NuisanceSample
from phoneme_modelling.util.acoustics import landmarks


def speech_ms(path: Path) -> float:
    samples, _ = sf.read(path)
    start, end = landmarks.active_span(samples, RATE)
    return 1000 * (end - start) / RATE


def sample(duration_ms: float) -> NuisanceSample:
    return NuisanceSample(duration_ms=duration_ms, rms_dbfs=-20.0, voice="kore")


@pytest.mark.parametrize("intensity", [0.0, 0.2, 0.4, 0.8])
def test_only_runs_at_the_supported_intensities(intensity: float, sound_files: dict[str, Path], tmp_path: Path) -> None:
    target = tmp_path / "out.wav"
    assert run(RToWMatchedDistorter(), sound_files["r"], intensity, target) == (False, None)
    assert not target.exists()


def test_error_clip_takes_the_length_of_a_correct_clip(sound_files: dict[str, Path], tmp_path: Path) -> None:
    distorter = MToNMatchedDistorter()
    distorter.parameters.correct_clips = [sample(1000.0)]
    distorter.parameters.error_checks = []
    plain, matched = tmp_path / "plain.wav", tmp_path / "matched.wav"
    run(MToNDistorter(), sound_files["m"], 1.0, plain)
    generated, ipa = run(distorter, sound_files["m"], 1.0, matched)
    assert generated
    assert ipa == "distortion[m(0%)|n(100%)]"
    assert speech_ms(matched) == pytest.approx(1000.0, rel=0.25)
    assert speech_ms(matched) > speech_ms(plain) * 1.3


def test_edit_that_keeps_damaging_the_sound_falls_back_to_the_original_length(
    sound_files: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def always_fails(*args: object, **kwargs: object) -> bool:
        calls.append(args)
        return False

    monkeypatch.setattr(pair_distorter, "guard", always_fails)
    distorter = MToNMatchedDistorter()
    distorter.parameters.correct_clips = [sample(1500.0)]
    target = tmp_path / "out.wav"
    generated, _ = run(distorter, sound_files["m"], 1.0, target)
    assert generated
    assert len(calls) == pair_distorter.MAX_GUARD_RETRIES + 1
    assert speech_ms(target) < 1000.0


def test_no_clips_to_match_against_is_rejected(sound_files: dict[str, Path], tmp_path: Path) -> None:
    distorter = RToWMatchedDistorter()
    distorter.parameters.correct_clips = []
    assert run(distorter, sound_files["r"], 1.0, tmp_path / "out.wav") == (False, None)
