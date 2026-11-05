from __future__ import annotations

import asyncio
import subprocess
import wave
from pathlib import Path
from typing import Optional

import pandas as pd
from google import genai
from google.genai.types import (Modality, Part, AudioTranscriptionConfig, SpeechConfig, VoiceConfig, 
                                PrebuiltVoiceConfig, LiveConnectConfig, Content)

from phoneme_modelling.schema.gemini_config import GeminiConfig
from phoneme_modelling.synthetic_audio.generation import (
    SyntheticAudioGenerationVoice,
    SyntheticAudioGenerator,
)


OPENING = (
    "You are helping me build a speech therapy teaching tool. I need very short, isolated speech sounds, "
    "not words. Never say the letter name and never add a vowel afterward unless asked. Keep the clip under "
    "2 seconds. Use a clear, natural voice at a relaxed pace, as if recorded close to a good microphone in a quiet "
    "room: no background noise, music, effects or breathing sounds. The clips will be used to train an audio model, "
    "so make the sound clean and easy to hear. Reply with only the sound."
)



class GeminiAudioVoice(SyntheticAudioGenerationVoice):

    def __init__(
        self,
        project: str,
        location: str,
        model: str,
        voice_name: str,
        prompts: dict,
        live: bool = False,
        ref_dir: Optional[Path] = None,
        ref_mode: str = "ref",
    ) -> None:
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.model = model
        self.voice_name = voice_name
        self.prompts = prompts
        self.live = live
        self.ref_dir = Path(ref_dir) if ref_dir else None
        self.ref_mode = ref_mode

    def get_voice_nickname(self) -> str:
        return self.voice_name.lower()

    def _load_ref(self, text: str, ipa: str) -> tuple[bytes, str] | None:
        if not self.ref_dir:
            return None
        target = text
        path = self.ref_dir / f"{target}.wav"
        if not path.exists():
            return None
        pcm = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
            capture_output=True, check=True,
        ).stdout
        if self.ref_mode == "mimic":
            return pcm, "Repeat exactly the sound in this recording, in your own voice, as clean and natural as possible. Don't copy the recording quality. Only the sound."
        
        intro = f"The audio is a recording of a person making the correct {target.replace('_', ' ').upper()} sound, IPA {ipa}. Use it only as a reference for which sound to make. Use your own voice, and make the sound as clean and natural as possible. Don't copy the recording's voice, recording quality, length, background noise, or any vowel before or after the sound. Then do the following."
        return pcm, f"{intro}\n\nMake the {target} sound."

    def __save_pcm_as_wav(self, path: Path, pcm: bytes, rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(pcm)

    async def __run_live_session(self, parts: list) -> tuple[bytes, str]:
        config = LiveConnectConfig(
            response_modalities=[Modality.AUDIO],
            system_instruction=OPENING,
            output_audio_transcription=AudioTranscriptionConfig(),
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=self.voice_name))
            ),
        )
        chunks, text = [], ""
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            await session.send_client_content(
                turns=Content(role="user", parts=parts), turn_complete=True
            )
            async for msg in session.receive():
                sc = msg.server_content
                if not sc:
                    continue
                if sc.output_transcription:
                    text += sc.output_transcription.text or ""
                if not sc.model_turn:
                    continue
                if not sc.model_turn.parts:
                    continue
                for part in sc.model_turn.parts:
                    if part.inline_data:
                        chunks.append(part.inline_data.data)
                if sc.turn_complete:
                    break
        return b"".join(chunks), text

    def __generate_live(self, prompt: str, ref: tuple[bytes, str] | None = None) -> tuple[bytes, int, str]:
        parts = [Part(text=prompt)]
        if ref:
            pcm_ref, text = ref
            parts = [
                Part.from_bytes(data=pcm_ref, mime_type="audio/pcm;rate=16000"),
                Part(text=text),
            ]

        pcm, text = asyncio.run(self.__run_live_session(parts))
        return pcm, 24000, text

    def __is_audio_too_short(self, pcm, rate) -> bool:
        return len(pcm) < 2 * rate // 20

    def _generate_audio(
        self,
        target_file_path: Path,
        text: str,
        ipa: str,
        distortion_intensity: Optional[float] = None,
        distort_to_text: Optional[str] = None,
        distort_to_ipa: Optional[str] = None,
    ) -> bool:
        prompt = f"Make the {text} sound (IPA: {ipa})."
        if text in self.prompts:
            prompt = self.prompts[text]
        ref = self._load_ref(text, ipa) if self.ref_dir else None
        pcm, rate, _ = self.__generate_live(prompt, ref)
        if self.__is_audio_too_short(pcm, rate):
            return False
        self.__save_pcm_as_wav(target_file_path, pcm, rate)
        return True


class GeminiAudioGenerator(SyntheticAudioGenerator):

    def __init__(
        self,
        config: GeminiConfig,
    ) -> None:
        self.project = config.project
        self.location = config.location
        self.model = config.model
        self.voice_names = config.voices
        self.live = config.live
        self.ref_dir = config.ref_dir
        self.ref_mode = config.ref_mode
        self.prompts_file = config.prompts_file
        super().__init__()

    def _init_voices(self) -> list[SyntheticAudioGenerationVoice]:
        prompts = {}
        if self.prompts_file is not None:
            data: list[dict] = pd.read_csv(self.prompts_file).to_dict(orient="records")
            prompts = {row["phoneme"]: row["prompt"] for row in data}
        return [
            GeminiAudioVoice(
                project=self.project,
                location=self.location,
                model=self.model,
                voice_name=voice_name,
                prompts=prompts,
                live=self.live,
                ref_dir=self.ref_dir,
                ref_mode=self.ref_mode,
            )
            for voice_name in self.voice_names
        ]

    def is_distortion_capable(self) -> bool:
        return False

    @classmethod
    def get_config_name(cls) -> str:
        return "gemini"

    def allows_retries(self) -> bool:
        """Returns whether the generation method is non-deterministic and can yield a different result if retried."""
        return True