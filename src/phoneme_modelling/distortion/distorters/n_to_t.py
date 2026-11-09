import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import nasal_to_stop
from phoneme_modelling.util.acoustics.clip import Clip


class NToTDistorter(PairDistorter):
    """n said as t: shortens and muffles the murmur into silence, then adds a t-like burst. Measured on 1 voice only."""

    FROM_PHONEME = "n"
    TO_PHONEME = "t"

    @staticmethod
    def get_name() -> str:
        return "n_to_t"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return nasal_to_stop(clip, alpha, self.parameters, rng, voiced=False)


class NToTMatchedDistorter(NToTDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "n_to_t_matched"
