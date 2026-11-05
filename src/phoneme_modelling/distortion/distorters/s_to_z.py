import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import voice_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class SToZDistorter(PairDistorter):
    """s said as z: reshapes the hiss, then adds voicing."""

    FROM_PHONEME = "s"
    TO_PHONEME = "z"

    @staticmethod
    def get_name() -> str:
        return "s_to_z"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return voice_fricative(clip, alpha, self.parameters, voice_f0, rng)


class SToZMatchedDistorter(SToZDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "s_to_z_matched"
