import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import affricate_to_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class JToShDistorter(PairDistorter):
    """j said as sh: fades the closure, softens the burst into a gradual onset and reshapes the hiss, and removes
    voicing. The 10-25 scorecard rates this pair "partly (overshoots)"."""

    FROM_PHONEME = "j"
    TO_PHONEME = "sh"

    @staticmethod
    def get_name() -> str:
        return "j_to_sh"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return affricate_to_fricative(clip, alpha, self.parameters, voice_f0, rng, voicing="off")


class JToShMatchedDistorter(JToShDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "j_to_sh_matched"
