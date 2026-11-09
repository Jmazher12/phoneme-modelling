import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import change_stop
from phoneme_modelling.util.acoustics.clip import Clip


class BToPDistorter(PairDistorter):
    """b said as p: fades the prevoicing and adds aspiration to the release. Burst place stays, it barely differs."""

    FROM_PHONEME = "b"
    TO_PHONEME = "p"

    @staticmethod
    def get_name() -> str:
        return "b_to_p"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return change_stop(clip, measures, alpha, self.parameters, voice_f0, rng, voicing="off", place=False)


class BToPMatchedDistorter(BToPDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "b_to_p_matched"
