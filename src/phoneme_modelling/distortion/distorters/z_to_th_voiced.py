import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import reshape_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class ZToThVoicedDistorter(PairDistorter):
    """z said as voiced th: reshapes the hiss; voicing stays as it is."""

    FROM_PHONEME = "z"
    TO_PHONEME = "th_voiced"

    @staticmethod
    def get_name() -> str:
        return "z_to_th_voiced"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return reshape_fricative(clip, alpha, self.parameters)


class ZToThVoicedMatchedDistorter(ZToThVoicedDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "z_to_th_voiced_matched"
