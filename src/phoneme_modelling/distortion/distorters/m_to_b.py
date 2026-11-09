import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import nasal_to_stop
from phoneme_modelling.util.acoustics.clip import Clip


class MToBDistorter(PairDistorter):
    """m said as b: shortens and muffles the murmur into a short voiced closure, then adds a b-like burst. The 10-25
    scorecard rates this pair "partly (overshoots)"."""

    FROM_PHONEME = "m"
    TO_PHONEME = "b"

    @staticmethod
    def get_name() -> str:
        return "m_to_b"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return nasal_to_stop(clip, alpha, self.parameters, rng, voiced=True)


class MToBMatchedDistorter(MToBDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "m_to_b_matched"
