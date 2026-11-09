import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import devoice_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class VToFDistorter(PairDistorter):
    """v said as f: removes voicing, then reshapes the hiss. The 10-25 scorecard rates this pair "partly (uneven
    steps)"."""

    FROM_PHONEME = "v"
    TO_PHONEME = "f"

    @staticmethod
    def get_name() -> str:
        return "v_to_f"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return devoice_fricative(clip, alpha, self.parameters, rng)


class VToFMatchedDistorter(VToFDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "v_to_f_matched"
