from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from conftest import RATE, seconds

from phoneme_modelling.distortion.distorter import AudioDistorter
from phoneme_modelling.distortion.distorters.m_to_n import MToNDistorter
from phoneme_modelling.distortion.distorters.p_to_b import PToBDistorter
from phoneme_modelling.distortion.distorters.r_to_w import RToWDistorter
from phoneme_modelling.distortion.distorters.s_to_z import SToZDistorter
from phoneme_modelling.distortion.distorters.th_voiced_to_th_voiceless import ThVoicedToThVoicelessDistorter
from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.pipeline import AudioDistortionPipeline
from phoneme_modelling.phonemes import TEXT_TO_IPA
from phoneme_modelling.schema.phoneme_pipeline_config import DistortionApplicationConfig, PhonemePipelineConfig
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.util.acoustics.measures import measure_as

PAIRS = [RToWDistorter, SToZDistorter, PToBDistorter, ThVoicedToThVoicelessDistorter, MToNDistorter]


def run(distorter: PairDistorter, source: Path, intensity: float, target: Path) -> tuple[bool, str | None]:
    return distorter.distort(
        original_file_path=source,
        original_phoneme_text=distorter.FROM_PHONEME,
        original_phoneme_ipa=TEXT_TO_IPA[distorter.FROM_PHONEME],
        distortion_intensity=intensity,
        distorted_file_path=target,
    )


@pytest.mark.parametrize("distorter_class", PAIRS)
def test_makes_a_standard_clip(
    distorter_class: type[PairDistorter], sound_files: dict[str, Path], tmp_path: Path
) -> None:
    distorter = distorter_class()
    target = tmp_path / "out.wav"
    generated, ipa = run(distorter, sound_files[distorter.FROM_PHONEME], 0.6, target)
    samples, rate = sf.read(target)
    assert generated
    assert rate == RATE
    assert len(samples) >= seconds(2.5)
    assert np.isfinite(samples).all()
    from_ipa, to_ipa = TEXT_TO_IPA[distorter.FROM_PHONEME], TEXT_TO_IPA[distorter.TO_PHONEME]
    assert ipa == f"distortion[{from_ipa}(40%)|{to_ipa}(60%)]"


@pytest.mark.parametrize("distorter_class", PAIRS)
def test_only_runs_on_its_own_sound_and_is_deterministic(distorter_class: type[PairDistorter]) -> None:
    distorter = distorter_class()
    assert distorter.is_restricted_to_phonemes() == [distorter.FROM_PHONEME]
    assert distorter.allows_retries() is False
    assert isinstance(distorter, AudioDistorter)


@pytest.mark.parametrize("distorter_class", PAIRS)
def test_intensity_maps_to_the_calibrated_strengths(distorter_class: type[PairDistorter]) -> None:
    distorter = distorter_class()
    calibrated = distorter.parameters.severity_alpha
    assert distorter.alpha_for(0.0) == 0.0
    assert distorter.alpha_for(0.2) == pytest.approx(calibrated[0])
    assert distorter.alpha_for(1.0) == pytest.approx(calibrated[4])
    assert distorter.alpha_for(0.3) == pytest.approx((calibrated[0] + calibrated[1]) / 2)


def test_intensity_zero_is_an_unedited_control(sound_files: dict[str, Path], tmp_path: Path) -> None:
    generated, ipa = run(RToWDistorter(), sound_files["r"], 0.0, tmp_path / "out.wav")
    assert generated
    assert ipa is None


def test_same_input_gives_the_same_audio_whatever_the_file_counter(
    sound_files: dict[str, Path], tmp_path: Path
) -> None:
    renamed = tmp_path / "123_r_kore.wav"
    renamed.write_bytes(sound_files["r"].read_bytes())
    run(RToWDistorter(), sound_files["r"], 0.6, tmp_path / "a.wav")
    run(RToWDistorter(), renamed, 0.6, tmp_path / "b.wav")
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


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


def test_r_to_w_raises_f3_and_lowers_f2() -> None:
    change = RToWDistorter().parameters.cue_change
    assert change["f3_hz"] > 0
    assert change["f2_hz"] < 0


def test_pipeline_makes_every_intensity_and_skips_other_sounds(sound_files: dict[str, Path], tmp_path: Path) -> None:
    rows = [
        SyntheticPhonemeMetadata(
            file_name=path.name,
            file_path=str(path),
            target_phoneme_text=key,
            actual_phoneme_text=key,
            actual_phoneme_ipa=TEXT_TO_IPA[key],
            distortion_intensity=0.0,
        ).model_dump()
        for key, path in sound_files.items()
        if key in ("r", "s")
    ]
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    config = PhonemePipelineConfig(
        number_of_digits=3,
        phonemes_to_generate=["r"],
        distortion_application_config=DistortionApplicationConfig(
            distortion_intensitites=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ),
        generator_configs={},
        distorter_configs={"r_to_w": None, "r_to_w_matched": None},
    )
    output = tmp_path / "phonemes"
    AudioDistortionPipeline.run(config, output, metadata_path, current_audio_count=2)

    metadata = pd.read_csv(metadata_path)
    made = metadata.iloc[2:]
    assert len(made) == 9  # 6 plain + 3 matched, and nothing for s
    assert all(Path(path).is_file() for path in made["file_path"])
    assert set(made["target_phoneme_text"]) == {"r"}
    assert (made["file_name"].str.contains("r_to_w_matched")).sum() == 3
