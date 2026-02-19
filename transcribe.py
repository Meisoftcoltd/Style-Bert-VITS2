import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset
from tqdm import tqdm

from config import get_path_config
from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


# HF pipelineで進捗表示をするために必要なDatasetクラス
class StrListDataset(Dataset[str]):
    def __init__(self, original_list: list[str]) -> None:
        self.original_list = original_list

    def __len__(self) -> int:
        return len(self.original_list)

    def __getitem__(self, i: int) -> str:
        return self.original_list[i]


# HFのWhisperはファイルリストを与えるとバッチ処理ができて速い
def transcribe_files_with_hf_whisper(
    audio_files: list[Path],
    model_id: str,
    output_file: Path,
    input_dir: Path,
    dataset_model_name: str,
    language_id: int,
    initial_prompt: Optional[str] = None,
    language: str = "ja",
    batch_size: int = 16,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 10,
    device: str = "cuda",
    pbar: Optional[tqdm] = None,
) -> list[str]:
    import torch
    from transformers import WhisperProcessor, pipeline

    generate_kwargs: dict[str, Any] = {
        "language": language,
        "do_sample": False,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    logger.info(f"generate_kwargs: {generate_kwargs}, loading pipeline...")
    pipe = pipeline(
        model=model_id,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=batch_size,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        trust_remote_code=True,
        # generate_kwargs=generate_kwargs,
    )
    logger.info("Loaded pipeline")
    if initial_prompt is not None:
        prompt_ids: torch.Tensor = pipe.tokenizer.encode(
            initial_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        generate_kwargs["prompt_ids"] = prompt_ids

    dataset = StrListDataset([str(f) for f in audio_files])

    results: list[str] = []
    for whisper_result, file in zip(
        pipe(dataset, generate_kwargs=generate_kwargs), audio_files
    ):
        text: str = whisper_result["text"]
        # なぜかテキストの最初に" {initial_prompt}"が入るので、文字の最初からこれを削除する
        # cf. https://github.com/huggingface/transformers/issues/27594
        if text.startswith(f" {initial_prompt}"):
            text = text[len(f" {initial_prompt}") :]

        with open(output_file, "a", encoding="utf-8") as f:
            wav_rel_path = file.relative_to(input_dir)
            f.write(f"{wav_rel_path}|{dataset_model_name}|{language_id}|{text}\n")
        results.append(text)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--initial_prompt",
        type=str,
        default="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
    )
    parser.add_argument(
        "--language", type=str, default="ja", choices=["ja", "en", "zh"]
    )
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_type", type=str, default="bfloat16")
    parser.add_argument("--use_hf_whisper", action="store_true", help="Deprecated, always true")
    parser.add_argument("--hf_repo_id", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=10)
    args = parser.parse_args()

    path_config = get_path_config()
    dataset_root = path_config.dataset_root

    dataset_model_name = str(args.model_name)

    input_dir = dataset_root / dataset_model_name / "raw"
    output_file = dataset_root / dataset_model_name / "esd.list"
    initial_prompt: str = args.initial_prompt
    initial_prompt = initial_prompt.strip('"')
    language: str = args.language
    device: str = args.device
    compute_type: str = args.compute_type
    batch_size: int = args.batch_size
    num_beams: int = args.num_beams
    no_repeat_ngram_size: int = args.no_repeat_ngram_size

    output_file.parent.mkdir(parents=True, exist_ok=True)

    wav_files = [f for f in input_dir.rglob("*.wav") if f.is_file()]
    wav_files = sorted(wav_files, key=lambda x: str(x))
    logger.info(f"Found {len(wav_files)} WAV files")
    if len(wav_files) == 0:
        logger.warning(f"No WAV files found in {input_dir}")
        sys.exit(1)

    if output_file.exists():
        logger.warning(f"{output_file} exists, backing up to {output_file}.bak")
        backup_path = output_file.with_name(output_file.name + ".bak")
        if backup_path.exists():
            logger.warning(f"{output_file}.bak exists, deleting...")
            backup_path.unlink()
        output_file.rename(backup_path)

    if language == "ja":
        language_id = Languages.JP.value
    elif language == "en":
        language_id = Languages.EN.value
    elif language == "zh":
        language_id = Languages.ZH.value
    else:
        raise ValueError(f"{language} is not supported.")

    # Model selection logic
    model_id = args.hf_repo_id
    if not model_id:
        if args.model == "large-v3":
            model_id = "openai/whisper-large-v3"
        elif args.model == "large-v2":
            model_id = "openai/whisper-large-v2"
        elif args.model == "large":
            model_id = "openai/whisper-large"
        elif args.model == "medium":
            model_id = "openai/whisper-medium"
        elif args.model == "small":
            model_id = "openai/whisper-small"
        elif args.model == "base":
            model_id = "openai/whisper-base"
        elif args.model == "tiny":
            model_id = "openai/whisper-tiny"
        else:
            model_id = args.model # Fallback
            logger.warning(f"Using {model_id} as HF model ID directly. Ensure this is a valid repo ID or path.")

    logger.info(f"Loading HF Whisper model ({model_id})")
    pbar = tqdm(total=len(wav_files), file=SAFE_STDOUT, dynamic_ncols=True)
    results = transcribe_files_with_hf_whisper(
        audio_files=wav_files,
        model_id=model_id,
        output_file=output_file,
        input_dir=input_dir,
        dataset_model_name=dataset_model_name,
        language_id=language_id,
        initial_prompt=initial_prompt,
        language=language,
        batch_size=batch_size,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        device=device,
        pbar=pbar,
    )

    sys.exit(0)
