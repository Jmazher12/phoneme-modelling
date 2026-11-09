import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import fricative_to_stop
from phoneme_modelling.util.acoustics.clip import Clip


class VToBDistorter(PairDistorter):
    """v said as b: makes the onset abrupt, cuts the hiss to a short release and shapes it like b's burst, then adds
    prevoicing. The 10-25 scorecard rates this pair "partly (overshoots)"."""

    FROM_PHONEME = "v"
    TO_PHONEME = "b"

    @staticmethod
    def get_name() -> str:
        return "v_to_b"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return fricative_to_stop(
            clip, alpha, self.parameters, voice_f0, rng, devoice=False, prevoice=True, shape="primary"
        )


class VToBMatchedDistorter(VToBDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "v_to_b_matched"
