from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer

from phoneme_modelling.util.ipa import clean_ipa
from phoneme_modelling.util.loaders import load_as_numpy_array
from phoneme_modelling.evaluation.evaluator import AudioEvaluator
from phoneme_modelling.phonemes import IPA_TO_TEXT
from phoneme_modelling.schema.phoneme_evaluation import PhonemeEvaluation


W2V2_ESPEAK_REPO = "facebook/wav2vec2-lv-60-espeak-cv-ft"
W2V2_ESPEAK_SAMPLE_RATE = 16000



class Wav2Vec2EspeakEvaluator(AudioEvaluator):
    """wav2vec2 trained on espeak IPA (Common Voice). Its phone set is closer to ours than TIMIT's (tʃ, dʒ, θ, ð)."""

    def __init__(self, config: Optional[BaseModel] = None):
        self.__feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(W2V2_ESPEAK_REPO)
        # do_phonemize=False: we only decode model output, so the espeak backend is not needed
        self.__tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(W2V2_ESPEAK_REPO, do_phonemize=False)
        self.__model = Wav2Vec2ForCTC.from_pretrained(W2V2_ESPEAK_REPO).eval()

    @staticmethod
    def get_name() -> str:
        return "wav2vec2_espeak"

    def evaluate(self, target_phoneme: str, target_phoneme_ipa: str, audio_path: Path) -> PhonemeEvaluation:
        audio = load_as_numpy_array(audio_path, W2V2_ESPEAK_SAMPLE_RATE)
        with torch.no_grad():
            encodings = self.__feature_extractor(audio, sampling_rate=W2V2_ESPEAK_SAMPLE_RATE, return_tensors="pt")
            logits = self.__model(encodings.input_values).logits
            detected_ipa = self.__tokenizer.batch_decode(logits.argmax(-1))[0].strip()
        detected_ipa = clean_ipa(detected_ipa)
        is_correct = detected_ipa == target_phoneme_ipa
        detected_text = None
        if detected_ipa in IPA_TO_TEXT:
            detected_text = IPA_TO_TEXT.get(detected_ipa)
        return PhonemeEvaluation(
            is_correct=is_correct,
            detected_phoneme_text=detected_text,
            detected_phoneme_ipa=detected_ipa,
        )
