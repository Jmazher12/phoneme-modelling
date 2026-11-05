"""OpenAI gpt-realtime* models through the Realtime API. Named openai_realtime.py so it can't shadow the openai package."""
import base64
import ssl
from pathlib import Path
from typing import Optional

import certifi
from openai import OpenAI

from phoneme_modelling.schema.openai_config import OpenAIConfig
from phoneme_modelling.synthetic_audio.generation import SyntheticAudioGenerationVoice, SyntheticAudioGenerator
from phoneme_modelling.synthetic_audio.llm_prompting import (
    OPENING,
    MissingReferenceError,
    PhonemePrompt,
    build_reference_message,
    load_prompts,
)
from phoneme_modelling.util.audio import peak, resample_pcm16, save_raw_and_prepped, save_wav, seconds
from phoneme_modelling.util.env import load_env


OPENAI_SAMPLE_RATE = 24000  # Realtime audio in and out is 24 kHz 16-bit mono PCM
MIN_SECONDS = 0.05  # shorter answers are effectively empty



class OpenAIAudioVoice(SyntheticAudioGenerationVoice):

    def __init__(
        self,
        client: OpenAI,
        model: str,
        voice_name: str,
        prompts: dict[str, PhonemePrompt],
        ref_dir: Path,
        take: int = 1,
        takes: int = 1,
        min_peak: float = 0.05,
    ) -> None:
        self.client = client
        self.model = model
        self.voice_name = voice_name
        self.prompts = prompts
        self.ref_dir = Path(ref_dir)
        self.take = take
        self.takes = takes
        self.min_peak = min_peak

    def get_voice_nickname(self) -> str:
        if self.takes > 1:
            return f"{self.voice_name.lower()}_take{self.take}"
        return self.voice_name.lower()

    def __run_realtime_session(self, reference_pcm: bytes, message: str) -> tuple[bytes, str]:
        """Realtime API: OPENING as the session instructions, then one user item with the reference recording
        (resampled to 24 kHz) and the message."""
        reference_24k = resample_pcm16(reference_pcm, 16000, OPENAI_SAMPLE_RATE)
        content = [
            {"type": "input_audio", "audio": base64.b64encode(reference_24k).decode()},
            {"type": "input_text", "text": message},
        ]
        chunks, transcript = [], ""
        ssl_context = ssl.create_default_context(cafile=certifi.where())  # the WebSocket uses Python's own SSL
        with self.client.realtime.connect(model=self.model, websocket_connection_options={"ssl": ssl_context}) as conn:
            conn.session.update(session={  # type: ignore[arg-type]
                "type": "realtime",
                "instructions": OPENING,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {"turn_detection": None},
                    "output": {"voice": self.voice_name, "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE}},
                },
            })
            conn.conversation.item.create(item={"type": "message", "role": "user", "content": content})  # type: ignore[arg-type]
            conn.response.create()
            for event in conn:
                if event.type == "response.output_audio.delta":
                    chunks.append(base64.b64decode(event.delta))
                elif event.type == "response.output_audio_transcript.delta":
                    transcript += event.delta
                elif event.type == "error":
                    raise RuntimeError(event.error.message)
                elif event.type == "response.done":
                    break
        return b"".join(chunks), transcript

    def _generate_audio(
        self,
        target_file_path: Path,
        text: str,
        ipa: str,
        distortion_intensity: Optional[float] = None,
        distort_to_text: Optional[str] = None,
        distort_to_ipa: Optional[str] = None,
    ) -> bool:
        self._generation_info = {}
        label = f"{target_file_path.stem} ({self.model}, {self.voice_name})"
        try:
            reference_pcm, message, reference_path = build_reference_message(text, self.prompts, self.ref_dir)
            pcm, transcript = self.__run_realtime_session(reference_pcm, message)
        except MissingReferenceError as e:
            print(f"FAILED {label}: {e}")
            return False
        except Exception as e:  # keep going: one failed take shouldn't stop the run
            print(f"FAILED {label}: {e}")
            return False
        rate = OPENAI_SAMPLE_RATE
        if seconds(pcm, rate) < MIN_SECONDS:
            print(f"FAILED {label}: model returned no audio")
            return False
        if peak(pcm) < self.min_peak:  # prep would boost the noise floor into static; keep the raw take for listening
            save_wav(target_file_path.parent / "raw" / f"{target_file_path.stem}_too_quiet.wav", pcm, rate)
            print(f"FAILED {label}: too quiet (peak {peak(pcm):.3f}), not prepped")
            return False
        raw_path, cropped = save_raw_and_prepped(pcm, rate, target_file_path)
        print(f"{label}: {seconds(pcm, rate):.1f}s, peak {peak(pcm):.2f}{', cropped' if cropped else ''}, model said {transcript!r}")
        self._generation_info = {
            "model": self.model,
            "voice": self.voice_name,
            "take": self.take,
            "reference_audio": str(reference_path),
            "prompt": message,
            "transcript": transcript,
            "raw_file_path": str(raw_path),
            "raw_seconds": round(seconds(pcm, rate), 2),
            "cropped": cropped,
        }
        return True


class OpenAIAudioGenerator(SyntheticAudioGenerator):

    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config
        super().__init__()

    def _init_voices(self) -> list[SyntheticAudioGenerationVoice]:
        config = self.config
        load_env(config.env_file)
        client = OpenAI()  # reads OPENAI_API_KEY
        prompts = load_prompts(config.prompts_file)
        return [
            OpenAIAudioVoice(
                client=client,
                model=config.model,
                voice_name=voice_name,
                prompts=prompts,
                ref_dir=config.ref_dir,
                take=take,
                takes=config.takes,
                min_peak=config.min_peak,
            )
            for voice_name in config.voices
            for take in range(1, config.takes + 1)
        ]

    def is_distortion_capable(self) -> bool:
        return False

    @classmethod
    def get_config_name(cls) -> str:
        return "openai"

    def allows_retries(self) -> bool:
        """Returns whether the generation method is non-deterministic and can yield a different result if retried."""
        return True
