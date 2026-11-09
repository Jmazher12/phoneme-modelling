from typing import Optional

import numpy as np
from pydantic import BaseModel


class CueCheck(BaseModel):
    """A defining cue that a nuisance edit must not move, with the natural gap between the two sounds."""

    cue: str
    natural_gap: float


class NuisanceSample(BaseModel):
    """Length, level and pitch of one clip, used as a target by the matched distorters."""

    duration_ms: float
    rms_dbfs: float
    f0_hz: Optional[float] = None
    voice: str


class PairParameters(BaseModel):
    """Frozen numbers for one sound pair, measured and calibrated on the 2026-10-25 clean clips.

    ``intensity_alpha`` maps each supported strength to the edit strength that reaches that share of the natural
    difference between the two sounds. Shapes are dB gains on ``grid_hz`` and are missing when the baseline had none.
    """

    from_phoneme: str
    to_phoneme: str
    pair_name: str
    family: str
    intensity_alpha: dict[float, float]
    cue_change: dict[str, float]
    target_value: dict[str, float]
    grid_hz: list[float]
    span_shape_db: Optional[list[float]] = None
    primary_shape_db: Optional[list[float]] = None
    target_primary_shape_db: Optional[list[float]] = None
    error_checks: list[CueCheck]
    correct_clips: list[NuisanceSample]

    def target(self, cue: str, default: float) -> float:
        """The produced sound's typical value of ``cue``."""
        return self.target_value.get(cue, default)

    def change(self, cue: str, default: float) -> float:
        """The natural change of ``cue`` from the source to the produced sound (log ratio for frequencies)."""
        return self.cue_change.get(cue, default)

    def shape(self, key: str) -> Optional[np.ndarray]:
        """``span_shape_db``, ``primary_shape_db`` or ``target_primary_shape_db`` as an array, if measured."""
        values = getattr(self, key)
        return np.asarray(values) if values else None
