import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import affricate_to_stop
from phoneme_modelling.util.acoustics.clip import Clip


class ChToTDistorter(PairDistorter):
    """ch said as t: cuts the hiss after the burst to a short release and reshapes the burst toward t's. The 10-25
    scorecard rates this pair "needs work"."""

    FROM_PHONEME = "ch"
    TO_PHONEME = "t"

    @staticmethod
    def get_name() -> str:
        return "ch_to_t"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return affricate_to_stop(clip, measures, alpha, self.parameters, voice_f0, rng, prevoice=False)


class ChToTMatchedDistorter(ChToTDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "ch_to_t_matched"
