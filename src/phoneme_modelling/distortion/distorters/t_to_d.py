import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import change_stop
from phoneme_modelling.util.acoustics.clip import Clip


class TToDDistorter(PairDistorter):
    """t said as d: adds prevoicing and cuts the release toward d, and reshapes the burst toward d's."""

    FROM_PHONEME = "t"
    TO_PHONEME = "d"

    @staticmethod
    def get_name() -> str:
        return "t_to_d"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return change_stop(clip, measures, alpha, self.parameters, voice_f0, rng, voicing="on", place=True)


class TToDMatchedDistorter(TToDDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "t_to_d_matched"
