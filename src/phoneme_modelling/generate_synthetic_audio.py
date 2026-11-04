import argparse
import json
import os
from pathlib import Path

from phoneme_modelling.distortion.pipeline import AudioDistortionPipeline
from phoneme_modelling.evaluation.pipeline import AudioEvaluationPipeline
from phoneme_modelling.schema.phoneme_pipeline_config import PhonemePipelineConfig
from phoneme_modelling.synthetic_audio.pipeline import SyntheticAudioPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/default.json")
    parser.add_argument("--output_folder", default="phonemes")
    parser.add_argument("--output_metadata_path", default="data/phonemes_metadata.csv")
    parser.add_argument("--evaluation_output_path", default="data/evaluation_results.csv")
    parser.add_argument("--mode", choices=["all", "generate", "eval"], default="all")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    output_folder = Path(args.output_folder)
    output_metadata_path = Path(args.output_metadata_path)
    evaluation_output_path = Path(args.evaluation_output_path)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_metadata_path.parent, exist_ok=True)
    os.makedirs(evaluation_output_path.parent, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = json.load(config_file)
        config = PhonemePipelineConfig.model_validate(config_dict)

    if args.mode in ["all", "generate"]:
        synthetic_audio_count = SyntheticAudioPipeline.run(config, output_folder, output_metadata_path)
        AudioDistortionPipeline.run(config, output_folder, output_metadata_path, synthetic_audio_count)
    if args.mode in ["all", "eval"]:
        AudioEvaluationPipeline.run(config, output_metadata_path, evaluation_output_path)

if __name__ == "__main__":
    main()