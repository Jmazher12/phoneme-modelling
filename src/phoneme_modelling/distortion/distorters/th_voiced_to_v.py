import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import reshape_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class ThVoicedToVDistorter(PairDistorter):
    """voiced th said as v: reshapes the hiss; voicing stays as it is. The 10-25 scorecard rates this pair
    "needs work"."""

    FROM_PHONEME = "th_voiced"
    TO_PHONEME = "v"

    @staticmethod
    def get_name() -> str:
        return "th_voiced_to_v"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return reshape_fricative(clip, alpha, self.parameters)


class ThVoicedToVMatchedDistorter(ThVoicedToVDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "th_voiced_to_v_matched"
