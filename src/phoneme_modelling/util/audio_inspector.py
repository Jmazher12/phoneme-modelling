from dataclasses import dataclass
from pathlib import Path

import wave



@dataclass
class AudioInfo:
    duration: float


class AudioInspector:

    @classmethod
    def get_info(cls, file_path: Path):
        duration = cls._get_wav_duration(file_path)
        return AudioInfo(
            duration=duration
        )

    @staticmethod
    def _get_wav_duration(file_path: Path) -> float:
        with wave.open(str(file_path), "rb") as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            return frames / float(rate)
