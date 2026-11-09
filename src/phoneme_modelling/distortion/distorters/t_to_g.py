import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import change_stop
from phoneme_modelling.util.acoustics.clip import Clip


class TToGDistorter(PairDistorter):
    """t said as g: adds prevoicing and cuts the release toward g, and reshapes the burst toward g's."""

    FROM_PHONEME = "t"
    TO_PHONEME = "g"

    @staticmethod
    def get_name() -> str:
        return "t_to_g"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return change_stop(clip, measures, alpha, self.parameters, voice_f0, rng, voicing="on", place=True)


class TToGMatchedDistorter(TToGDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "t_to_g_matched"
