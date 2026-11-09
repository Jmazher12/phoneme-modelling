import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import fricative_to_stop
from phoneme_modelling.util.acoustics.clip import Clip


class ShToChDistorter(PairDistorter):
    """sh said as ch: makes the onset abrupt and cuts the hiss toward ch's length."""

    FROM_PHONEME = "sh"
    TO_PHONEME = "ch"

    @staticmethod
    def get_name() -> str:
        return "sh_to_ch"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return fricative_to_stop(
            clip, alpha, self.parameters, voice_f0, rng, devoice=False, prevoice=False, shape="span"
        )


class ShToChMatchedDistorter(ShToChDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "sh_to_ch_matched"
