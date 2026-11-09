import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import fricative_to_stop
from phoneme_modelling.util.acoustics.clip import Clip


class SToDDistorter(PairDistorter):
    """s said as d: makes the onset abrupt, cuts the hiss to a short release and shapes it like d's burst, then adds
    prevoicing. The 10-25 scorecard rates this pair "partly (overshoots)"."""

    FROM_PHONEME = "s"
    TO_PHONEME = "d"

    @staticmethod
    def get_name() -> str:
        return "s_to_d"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return fricative_to_stop(
            clip, alpha, self.parameters, voice_f0, rng, devoice=False, prevoice=True, shape="primary"
        )


class SToDMatchedDistorter(SToDDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "s_to_d_matched"
