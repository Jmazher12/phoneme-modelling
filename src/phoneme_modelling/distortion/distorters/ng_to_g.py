import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import nasal_to_stop
from phoneme_modelling.util.acoustics.clip import Clip


class NgToGDistorter(PairDistorter):
    """ng said as g: shortens and muffles the murmur into a short voiced closure, then adds a g-like burst. Measured on
    2 voices only; the 10-25 scorecard rates this pair "needs work"."""

    FROM_PHONEME = "ng"
    TO_PHONEME = "g"

    @staticmethod
    def get_name() -> str:
        return "ng_to_g"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return nasal_to_stop(clip, alpha, self.parameters, rng, voiced=True)


class NgToGMatchedDistorter(NgToGDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "ng_to_g_matched"
