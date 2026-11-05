import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import voice_stop
from phoneme_modelling.util.acoustics.clip import Clip


class PToBDistorter(PairDistorter):
    """p said as b: adds prevoicing and cuts the release toward b. Burst place stays, it barely differs."""

    FROM_PHONEME = "p"
    TO_PHONEME = "b"

    @staticmethod
    def get_name() -> str:
        return "p_to_b"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return voice_stop(clip, measures, alpha, self.parameters, voice_f0, rng)


class PToBMatchedDistorter(PToBDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "p_to_b_matched"
