from typing import Optional

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
    """Frozen numbers for one sound pair, measured and calibrated on the 2026-10-25 clean clips."""

    from_phoneme: str
    to_phoneme: str
    severity_alpha: list[float]
    cue_change: dict[str, float]
    target_value: dict[str, float]
    grid_hz: list[float]
    span_shape_db: list[float]
    error_checks: list[CueCheck]
    correct_checks: list[CueCheck]
    correct_clips: list[NuisanceSample]
    error_clips: list[NuisanceSample]

    def target(self, cue: str, default: float) -> float:
        return self.target_value.get(cue, default)
