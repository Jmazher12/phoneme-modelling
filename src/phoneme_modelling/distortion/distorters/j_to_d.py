import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import affricate_to_stop
from phoneme_modelling.util.acoustics.clip import Clip


class JToDDistorter(PairDistorter):
    """j said as d: cuts the hiss after the burst to a short release and reshapes the burst toward d's, then adds
    prevoicing."""

    FROM_PHONEME = "j"
    TO_PHONEME = "d"

    @staticmethod
    def get_name() -> str:
        return "j_to_d"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return affricate_to_stop(clip, measures, alpha, self.parameters, voice_f0, rng, prevoice=True)


class JToDMatchedDistorter(JToDDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "j_to_d_matched"
