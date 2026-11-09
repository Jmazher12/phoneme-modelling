import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import fricative_to_stop
from phoneme_modelling.util.acoustics.clip import Clip


class FToPDistorter(PairDistorter):
    """f said as p: makes the onset abrupt, cuts the hiss to a short release and shapes it like p's burst."""

    FROM_PHONEME = "f"
    TO_PHONEME = "p"

    @staticmethod
    def get_name() -> str:
        return "f_to_p"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return fricative_to_stop(
            clip, alpha, self.parameters, voice_f0, rng, devoice=False, prevoice=False, shape="primary"
        )


class FToPMatchedDistorter(FToPDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "f_to_p_matched"
