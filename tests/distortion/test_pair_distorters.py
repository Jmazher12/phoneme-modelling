import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from conftest import RATE, seconds

from phoneme_modelling.distortion.distorter import AudioDistorter
from phoneme_modelling.distortion.distorters.r_distortion import RDistortionDistorter
from phoneme_modelling.distortion.distorters.r_to_w import RToWDistorter
from phoneme_modelling.distortion.distorters.s_to_z import SToZDistorter
from phoneme_modelling.distortion.pair_distorter import INTENSITIES, PARAMETERS_FOLDER, PairDistorter, voice_of
from phoneme_modelling.distortion.pair_registry import PAIR_DISTORTERS
from phoneme_modelling.distortion.pipeline import AudioDistortionPipeline
from phoneme_modelling.phonemes import TEXT_TO_IPA
from phoneme_modelling.schema.phoneme_pipeline_config import (
    DistortionApplicationConfig,
    PhonemePipelineConfig,
    PhonemeValidationConfig,
)
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.util.acoustics import landmarks
from phoneme_modelling.util.acoustics.measures import measure_as

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.json"
PLAIN = [cls for cls in PAIR_DISTORTERS if not cls.MATCH_NUISANCE]
BY_NAME = {cls.get_name(): cls for cls in PAIR_DISTORTERS}


def run(distorter: PairDistorter, source: Path, intensity: float, target: Path) -> tuple[bool, str | None]:
    return distorter.distort(
        original_file_path=source,
        original_phoneme_text=distorter.FROM_PHONEME,
        original_phoneme_ipa=TEXT_TO_IPA[distorter.FROM_PHONEME],
        distortion_intensity=intensity,
        distorted_file_path=target,
    )


def distort(name: str, sound_files: dict[str, Path], tmp_path: Path, intensity: float = 1.0) -> tuple[dict, dict]:
    """Measures of the source clip (as its own sound) and of the distorted clip (as the produced sound)."""
    distorter = BY_NAME[name]()
    target = tmp_path / f"{name}.wav"
    assert run(distorter, sound_files[distorter.FROM_PHONEME], intensity, target)[0]
    before = measure_as(sf.read(sound_files[distorter.FROM_PHONEME])[0], RATE, distorter.FROM_PHONEME)
    after = measure_as(sf.read(target)[0], RATE, distorter.TO_PHONEME)
    return before, after


def speech_ms(samples: np.ndarray) -> float:
    start, end = landmarks.active_span(samples, RATE)
    return 1000 * (end - start) / RATE


def test_every_pair_has_a_plain_and_a_matched_distorter() -> None:
    names = [cls.get_name() for cls in PAIR_DISTORTERS]
    assert len(PLAIN) == 57
    assert len(names) == len(set(names)) == 114
    assert {f"{cls.get_name()}_matched" for cls in PLAIN} == set(names) - {cls.get_name() for cls in PLAIN}


def test_default_config_lists_every_distorter_at_the_two_strengths() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["distortion_application_config"]["distortion_intensitites"] == list(INTENSITIES) == [0.6, 1.0]
    assert set(config["distorter_configs"]) == {cls.get_name() for cls in PAIR_DISTORTERS}


def test_every_settings_file_belongs_to_a_distorter() -> None:
    files = {path.stem for path in PARAMETERS_FOLDER.glob("*.json")} - {"voice_f0"}
    assert files == {cls.get_name() for cls in PLAIN}


@pytest.mark.parametrize("distorter_class", PLAIN, ids=lambda cls: cls.get_name())
@pytest.mark.parametrize("intensity", INTENSITIES)
def test_makes_a_standard_clip(
    distorter_class: type[PairDistorter], intensity: float, sound_files: dict[str, Path], tmp_path: Path
) -> None:
    distorter = distorter_class()
    target = tmp_path / "out.wav"
    generated, ipa = run(distorter, sound_files[distorter.FROM_PHONEME], intensity, target)
    samples, rate = sf.read(target)
    assert generated
    assert rate == RATE
    assert len(samples) >= seconds(2.5)
    assert np.isfinite(samples).all()
    from_ipa, to_ipa = TEXT_TO_IPA[distorter.FROM_PHONEME], distorter.TO_IPA or TEXT_TO_IPA[distorter.TO_PHONEME]
    to_percentage = round(intensity * 100)
    assert ipa == f"distortion[{from_ipa}({100 - to_percentage}%)|{to_ipa}({to_percentage}%)]"


@pytest.mark.parametrize("distorter_class", PAIR_DISTORTERS, ids=lambda cls: cls.get_name())
def test_only_runs_on_its_own_sound_and_is_deterministic(distorter_class: type[PairDistorter]) -> None:
    distorter = distorter_class()
    assert distorter.is_restricted_to_phonemes() == [distorter.FROM_PHONEME]
    assert distorter.allows_retries() is False
    assert isinstance(distorter, AudioDistorter)


@pytest.mark.parametrize("distorter_class", PLAIN, ids=lambda cls: cls.get_name())
def test_intensity_maps_to_the_calibrated_strengths(distorter_class: type[PairDistorter]) -> None:
    distorter = distorter_class()
    calibrated = distorter.parameters.intensity_alpha
    assert set(calibrated) == set(INTENSITIES)
    assert distorter.alpha_for(0.6) == pytest.approx(calibrated[0.6])
    assert distorter.alpha_for(1.0) == pytest.approx(calibrated[1.0])
    for unsupported in (0.0, 0.2, 0.4, 0.8):
        assert distorter.alpha_for(unsupported) is None


@pytest.mark.parametrize("intensity", [0.0, 0.2, 0.4, 0.8])
def test_other_intensities_are_skipped(intensity: float, sound_files: dict[str, Path], tmp_path: Path) -> None:
    target = tmp_path / "out.wav"
    assert run(RToWDistorter(), sound_files["r"], intensity, target) == (False, None)
    assert not target.exists()


def test_distorted_r_is_labelled_as_a_lowered_r(sound_files: dict[str, Path], tmp_path: Path) -> None:
    distorter = RDistortionDistorter()
    assert distorter.parameters.pair_name == "r_distortion"
    assert run(distorter, sound_files["r"], 0.6, tmp_path / "out.wav") == (True, "distortion[ɹ(40%)|ɹ̞(60%)]")


def test_same_input_gives_the_same_audio_whatever_the_file_counter(
    sound_files: dict[str, Path], tmp_path: Path
) -> None:
    renamed = tmp_path / "123_r_kore.wav"
    renamed.write_bytes(sound_files["r"].read_bytes())
    run(RToWDistorter(), sound_files["r"], 0.6, tmp_path / "a.wav")
    run(RToWDistorter(), renamed, 0.6, tmp_path / "b.wav")
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


@pytest.mark.parametrize("seed_name", ["r_kore", "th_voiced_kore", "r_Kore_take2", "th_voiced_kore_take12"])
def test_voice_is_read_from_the_clip_name_with_or_without_a_take(seed_name: str) -> None:
    assert voice_of(seed_name) == "kore"


def test_silent_recording_is_rejected(tmp_path: Path) -> None:
    silent = tmp_path / "000_r_kore.wav"
    sf.write(silent, np.zeros(seconds(1.0)), RATE, subtype="PCM_16")
    target = tmp_path / "out.wav"
    assert run(RToWDistorter(), silent, 0.6, target) == (False, None)
    assert not target.exists()


def test_s_to_z_adds_voicing(sound_files: dict[str, Path], tmp_path: Path) -> None:
    target = tmp_path / "out.wav"
    run(SToZDistorter(), sound_files["s"], 1.0, target)
    before = measure_as(sf.read(sound_files["s"])[0], RATE, "s")
    after = measure_as(sf.read(target)[0], RATE, "z")
    assert after["voicebar_db"] > before["voicebar_db"] + 20


def test_z_to_t_removes_voicing(sound_files: dict[str, Path], tmp_path: Path) -> None:
    before, after = distort("z_to_t", sound_files, tmp_path)
    assert after["voicebar_db"] < before["voicebar_db"] - 20


def test_j_to_z_strengthens_voicing(sound_files: dict[str, Path], tmp_path: Path) -> None:
    before, after = distort("j_to_z", sound_files, tmp_path)
    assert after["voicebar_db"] > before["voicebar_db"] + 5


def test_r_to_w_raises_f3_and_lowers_f2() -> None:
    change = RToWDistorter().parameters.cue_change
    assert change["f3_hz"] > 0
    assert change["f2_hz"] < 0


def test_b_to_p_adds_aspiration(sound_files: dict[str, Path], tmp_path: Path) -> None:
    before, after = distort("b_to_p", sound_files, tmp_path)
    assert after["vot_ms"] > before["vot_ms"] + 15


def test_t_to_d_adds_prevoicing_and_shortens_the_release(sound_files: dict[str, Path], tmp_path: Path) -> None:
    before, after = distort("t_to_d", sound_files, tmp_path)
    assert after["closure_voicing_ms"] > before["closure_voicing_ms"] + 30
    assert after["vot_ms"] < before["vot_ms"] / 2


@pytest.mark.parametrize("name", ["s_to_t", "ch_to_t", "n_to_d"])
def test_manner_changes_to_a_stop_shorten_the_sound(name: str, sound_files: dict[str, Path], tmp_path: Path) -> None:
    distorter = BY_NAME[name]()
    target = tmp_path / "out.wav"
    run(distorter, sound_files[distorter.FROM_PHONEME], 1.0, target)
    assert speech_ms(sf.read(target)[0]) < 0.5 * speech_ms(sf.read(sound_files[distorter.FROM_PHONEME])[0])


def test_pipeline_makes_both_intensities_and_skips_other_sounds(sound_files: dict[str, Path], tmp_path: Path) -> None:
    rows = [
        SyntheticPhonemeMetadata(
            file_name=path.name,
            file_path=str(path),
            model="gemini",
            voice="kore",
            target_phoneme_text=key,
            actual_phoneme_text=key,
            actual_phoneme_ipa=TEXT_TO_IPA[key],
            distortion_intensity=0.0,
            duration=1.0,
        ).model_dump()
        for key, path in sound_files.items()
        if key in ("r", "s")
    ]
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    config = PhonemePipelineConfig(
        number_of_digits=3,
        phonemes_to_generate=["r"],
        distortion_application_config=DistortionApplicationConfig(distortion_intensitites=list(INTENSITIES)),
        validation_config=PhonemeValidationConfig(),
        generator_configs={},
        distorter_configs={"r_to_w": None, "r_to_w_matched": None},
    )
    output = tmp_path / "phonemes"
    AudioDistortionPipeline.run(config, output, metadata_path, current_audio_count=2)

    metadata = pd.read_csv(metadata_path)
    made = metadata.iloc[2:]
    assert len(made) == 4  # 2 plain + 2 matched, and nothing for s
    assert all(Path(path).is_file() for path in made["file_path"])
    assert set(made["target_phoneme_text"]) == {"r"}
    assert sorted(made["distortion_intensity"]) == [0.6, 0.6, 1.0, 1.0]
    assert (made["file_name"].str.contains("r_to_w_matched")).sum() == 2
