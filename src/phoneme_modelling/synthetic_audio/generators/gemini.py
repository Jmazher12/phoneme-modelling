import asyncio
from pathlib import Path
from typing import Optional

from google import genai
from google.genai.types import (Modality, Part, AudioTranscriptionConfig, SpeechConfig, VoiceConfig,
                                PrebuiltVoiceConfig, LiveConnectConfig, Content)

from phoneme_modelling.schema.gemini_config import GeminiConfig
from phoneme_modelling.synthetic_audio.generation import SyntheticAudioGenerationVoice, SyntheticAudioGenerator
from phoneme_modelling.synthetic_audio.llm_prompting import (
    OPENING,
    MissingReferenceError,
    PhonemePrompt,
    build_reference_message,
    load_prompts,
)
from phoneme_modelling.util.audio import save_raw_and_prepped, seconds


GEMINI_SAMPLE_RATE = 24000  # Live API output is 24 kHz 16-bit mono PCM
MIN_SECONDS = 0.05  # shorter answers are effectively empty



class GeminiAudioVoice(SyntheticAudioGenerationVoice):

    def __init__(
        self,
        client: genai.Client,
        model: str,
        voice_name: str,
        prompts: dict[str, PhonemePrompt],
        ref_dir: Path,
        take: int = 1,
        takes: int = 1,
    ) -> None:
        self.client = client
        self.model = model
        self.voice_name = voice_name
        self.prompts = prompts
        self.ref_dir = Path(ref_dir)
        self.take = take
        self.takes = takes

    def get_voice_nickname(self) -> str:
        if self.takes > 1:
            return f"{self.voice_name.lower()}_take{self.take}"
        return self.voice_name.lower()

    async def __run_live_session(self, reference_pcm: bytes, message: str) -> tuple[bytes, str]:
        """Live API: OPENING as the system instruction, then one turn with the reference recording and the message."""
        config = LiveConnectConfig(
            response_modalities=[Modality.AUDIO],
            system_instruction=OPENING,
            output_audio_transcription=AudioTranscriptionConfig(),
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=self.voice_name))
            ),
        )
        parts = [Part.from_bytes(data=reference_pcm, mime_type="audio/pcm;rate=16000"), Part(text=message)]
        chunks, transcript = [], ""
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            await session.send_client_content(turns=Content(role="user", parts=parts), turn_complete=True)
            async for msg in session.receive():
                sc = msg.server_content
                if not sc:
                    continue
                if sc.output_transcription:
                    transcript += sc.output_transcription.text or ""
                if sc.model_turn and sc.model_turn.parts:
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            chunks.append(part.inline_data.data)
                if sc.turn_complete:
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
            pcm, transcript = asyncio.run(self.__run_live_session(reference_pcm, message))
        except MissingReferenceError as e:
            print(f"FAILED {label}: {e}")
            return False
        except Exception as e:  # keep going: one failed take shouldn't stop the run
            print(f"FAILED {label}: {e}")
            return False
        rate = GEMINI_SAMPLE_RATE
        if seconds(pcm, rate) < MIN_SECONDS:
            print(f"FAILED {label}: model returned no audio")
            return False
        raw_path, cropped = save_raw_and_prepped(pcm, rate, target_file_path)
        print(f"{label}: {seconds(pcm, rate):.1f}s{', cropped' if cropped else ''}, model said {transcript!r}")
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


class GeminiAudioGenerator(SyntheticAudioGenerator):

    def __init__(self, config: GeminiConfig) -> None:
        self.config = config
        super().__init__()

    def _init_voices(self) -> list[SyntheticAudioGenerationVoice]:
        config = self.config
        client = genai.Client(vertexai=True, project=config.project, location=config.location)
        prompts = load_prompts(config.prompts_file)
        return [
            GeminiAudioVoice(
                client=client,
                model=config.model,
                voice_name=voice_name,
                prompts=prompts,
                ref_dir=config.ref_dir,
                take=take,
                takes=config.takes,
            )
            for voice_name in config.voices
            for take in range(1, config.takes + 1)
        ]

    def is_distortion_capable(self) -> bool:
        return False

    @classmethod
    def get_config_name(cls) -> str:
        return "gemini"

    def allows_retries(self) -> bool:
        """Returns whether the generation method is non-deterministic and can yield a different result if retried."""
        return True
