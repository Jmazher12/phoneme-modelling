import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import devoice_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class ZToSDistorter(PairDistorter):
    """z said as s: removes voicing, then reshapes the hiss."""

    FROM_PHONEME = "z"
    TO_PHONEME = "s"

    @staticmethod
    def get_name() -> str:
        return "z_to_s"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return devoice_fricative(clip, alpha, self.parameters, rng)


class ZToSMatchedDistorter(ZToSDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "z_to_s_matched"
