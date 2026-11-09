import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import change_nasal_place
from phoneme_modelling.util.acoustics.clip import Clip


class NgToNDistorter(PairDistorter):
    """ng said as n: reshapes the murmur. No voice had both sounds, so it compares all ng clips with all n clips."""

    FROM_PHONEME = "ng"
    TO_PHONEME = "n"

    @staticmethod
    def get_name() -> str:
        return "ng_to_n"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return change_nasal_place(clip, alpha, self.parameters)


class NgToNMatchedDistorter(NgToNDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "ng_to_n_matched"
