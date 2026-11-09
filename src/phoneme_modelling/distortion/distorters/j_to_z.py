import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import affricate_to_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class JToZDistorter(PairDistorter):
    """j said as z: fades the closure, softens the burst into a gradual onset and reshapes the hiss, and strengthens the
    voicing."""

    FROM_PHONEME = "j"
    TO_PHONEME = "z"

    @staticmethod
    def get_name() -> str:
        return "j_to_z"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return affricate_to_fricative(clip, alpha, self.parameters, voice_f0, rng, voicing="stronger")


class JToZMatchedDistorter(JToZDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "j_to_z_matched"
