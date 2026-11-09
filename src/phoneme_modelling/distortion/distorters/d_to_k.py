import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import change_stop
from phoneme_modelling.util.acoustics.clip import Clip


class DToKDistorter(PairDistorter):
    """d said as k: fades the prevoicing and adds aspiration to the release, and reshapes the burst toward k's."""

    FROM_PHONEME = "d"
    TO_PHONEME = "k"

    @staticmethod
    def get_name() -> str:
        return "d_to_k"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return change_stop(clip, measures, alpha, self.parameters, voice_f0, rng, voicing="off", place=True)


class DToKMatchedDistorter(DToKDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "d_to_k_matched"
