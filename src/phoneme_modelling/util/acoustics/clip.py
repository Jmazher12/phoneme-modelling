from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

# Label given to a vowel that follows the target sound.
VOWEL_LABEL = "ə"


@dataclass(frozen=True)
class Segment:
    phone: str
    start: float
    end: float


@dataclass
class Clip:
    """A recording plus where its sounds sit inside it. Segment times are seconds from the start of the clip."""

    samples: np.ndarray
    sample_rate: int
    segments: list[Segment]

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def bounds(self, index: int) -> tuple[int, int]:
        """Sample range of segment ``index``."""
        segment = self.segments[index]
        start = int(round(segment.start * self.sample_rate))
        end = int(round(segment.end * self.sample_rate))
        return max(0, start), min(len(self.samples), end)

    def speech_bounds(self) -> tuple[int, int]:
        """Sample range from the first segment's start to the last segment's end."""
        return self.bounds(0)[0], self.bounds(len(self.segments) - 1)[1]

    def segment_samples(self, index: int) -> np.ndarray:
        start, end = self.bounds(index)
        return self.samples[start:end]

    def is_vowel(self, index: int) -> bool:
        return 0 <= index < len(self.segments) and self.segments[index].phone == VOWEL_LABEL

    def with_samples(self, samples: np.ndarray, segments: Optional[list[Segment]] = None) -> "Clip":
        return replace(self, samples=samples, segments=self.segments if segments is None else segments)
