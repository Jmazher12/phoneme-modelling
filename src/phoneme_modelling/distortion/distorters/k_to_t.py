import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import change_stop
from phoneme_modelling.util.acoustics.clip import Clip


class KToTDistorter(PairDistorter):
    """k said as t: reshapes the burst toward t's."""

    FROM_PHONEME = "k"
    TO_PHONEME = "t"

    @staticmethod
    def get_name() -> str:
        return "k_to_t"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return change_stop(clip, measures, alpha, self.parameters, voice_f0, rng, voicing=None, place=True)


class KToTMatchedDistorter(KToTDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "k_to_t_matched"
