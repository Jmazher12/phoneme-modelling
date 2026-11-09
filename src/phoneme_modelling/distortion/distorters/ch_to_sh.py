import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import affricate_to_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class ChToShDistorter(PairDistorter):
    """ch said as sh: fades the closure, softens the burst into a gradual onset and reshapes the hiss. The 10-25
    scorecard rates this pair "needs work"."""

    FROM_PHONEME = "ch"
    TO_PHONEME = "sh"

    @staticmethod
    def get_name() -> str:
        return "ch_to_sh"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return affricate_to_fricative(clip, alpha, self.parameters, voice_f0, rng, voicing=None)


class ChToShMatchedDistorter(ChToShDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "ch_to_sh_matched"
