import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import reshape_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class ThVoicelessToFDistorter(PairDistorter):
    """voiceless th said as f: reshapes the hiss; voicing stays as it is. Measured on 1 voice only; the 10-25 scorecard
    rates this pair "needs work"."""

    FROM_PHONEME = "th_voiceless"
    TO_PHONEME = "f"

    @staticmethod
    def get_name() -> str:
        return "th_voiceless_to_f"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return reshape_fricative(clip, alpha, self.parameters)


class ThVoicelessToFMatchedDistorter(ThVoicelessToFDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "th_voiceless_to_f_matched"
