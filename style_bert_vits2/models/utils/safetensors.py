from pathlib import Path
from typing import Any, Optional, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.symbols import (
    ZH_SYMBOLS,
    JP_SYMBOLS,
    EN_SYMBOLS,
    PAD,
    PUNCTUATION_SYMBOLS,
    SYMBOLS,
)


def load_safetensors(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    for_infer: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.nn.Module, Optional[int]]:
    """
    指定されたパスから safetensors モデルを読み込み、モデルとイテレーションを返す。

    Args:
        checkpoint_path (Union[str, Path]): モデルのチェックポイントファイルのパス
        model (torch.nn.Module): 読み込む対象のモデル
        for_infer (bool): 推論用に読み込むかどうかのフラグ

    Returns:
        tuple[torch.nn.Module, Optional[int]]: 読み込まれたモデルとイテレーション回数（存在する場合）
    """

    tensors: dict[str, Any] = {}
    iteration: Optional[int] = None
    with safe_open(str(checkpoint_path), framework="pt", device=device) as f:  # type: ignore
        for key in f.keys():
            if key == "iteration":
                iteration = f.get_tensor(key).item()
            tensors[key] = f.get_tensor(key)

    # Check for JP-Extra (112 symbols) to Spanish-Extra (122 symbols) mismatch
    if "enc_p.emb.weight" in tensors:
        loaded_emb_shape = tensors["enc_p.emb.weight"].shape

        # Access the real model if wrapped (e.g. DataParallel)
        real_model = model.module if hasattr(model, "module") else model

        # We need to check if enc_p exists (it should for SynthesizerTrn)
        if hasattr(real_model, "enc_p"):
            expected_emb_shape = real_model.enc_p.emb.weight.shape
            expected_tone_shape = real_model.enc_p.tone_emb.weight.shape
            expected_lang_shape = real_model.enc_p.language_emb.weight.shape

            current_emb_weight = real_model.enc_p.emb.weight
            current_tone_weight = real_model.enc_p.tone_emb.weight
            current_lang_weight = real_model.enc_p.language_emb.weight

            if loaded_emb_shape != expected_emb_shape:
                # Reconstruct old symbols (assuming standard JP-Extra configuration: ZH+JP+EN)
                old_normal = sorted(set(ZH_SYMBOLS + JP_SYMBOLS + EN_SYMBOLS))
                old_symbols = [PAD] + old_normal + PUNCTUATION_SYMBOLS

                if loaded_emb_shape[0] == len(old_symbols):
                    logger.info(f"Adapting enc_p.emb.weight from old checkpoint format ({loaded_emb_shape[0]} -> {expected_emb_shape[0]})...")

                    loaded_weight = tensors["enc_p.emb.weight"]
                    # Create new weight tensor initialized from current model (random or pre-trained)
                    # Ensure it is on the same device as loaded_weight
                    new_emb_weight = current_emb_weight.to(loaded_weight.device).clone()

                    # Map old to new
                    mapped_count = 0
                    for i, sym in enumerate(old_symbols):
                        if sym in SYMBOLS:
                            new_idx = SYMBOLS.index(sym)
                            new_emb_weight[new_idx] = loaded_weight[i]
                            mapped_count += 1
                        else:
                            logger.warning(f"Symbol {sym} from checkpoint not found in current SYMBOLS!")

                    logger.info(f"Mapped {mapped_count} symbols.")
                    tensors["enc_p.emb.weight"] = new_emb_weight
                else:
                    logger.warning(f"Loaded enc_p.emb.weight shape {loaded_emb_shape} does not match expected {expected_emb_shape} and is not recognized as standard JP-Extra format (112). Skipping adaptation.")

            # Adapt tone embeddings
            if "enc_p.tone_emb.weight" in tensors:
                loaded_tone_shape = tensors["enc_p.tone_emb.weight"].shape
                if loaded_tone_shape != expected_tone_shape:
                    if loaded_tone_shape[0] < expected_tone_shape[0]:
                        logger.info(f"Adapting enc_p.tone_emb.weight from old checkpoint format ({loaded_tone_shape[0]} -> {expected_tone_shape[0]})...")
                        new_tone_weight = current_tone_weight.to(tensors["enc_p.tone_emb.weight"].device).clone()
                        new_tone_weight[:loaded_tone_shape[0]] = tensors["enc_p.tone_emb.weight"]
                        tensors["enc_p.tone_emb.weight"] = new_tone_weight

            # Adapt language embeddings
            if "enc_p.language_emb.weight" in tensors:
                loaded_lang_shape = tensors["enc_p.language_emb.weight"].shape
                if loaded_lang_shape != expected_lang_shape:
                    if loaded_lang_shape[0] < expected_lang_shape[0]:
                        logger.info(f"Adapting enc_p.language_emb.weight from old checkpoint format ({loaded_lang_shape[0]} -> {expected_lang_shape[0]})...")
                        new_lang_weight = current_lang_weight.to(tensors["enc_p.language_emb.weight"].device).clone()
                        new_lang_weight[:loaded_lang_shape[0]] = tensors["enc_p.language_emb.weight"]
                        tensors["enc_p.language_emb.weight"] = new_lang_weight

    if hasattr(model, "module"):
        result = model.module.load_state_dict(tensors, strict=False)
    else:
        result = model.load_state_dict(tensors, strict=False)
    for key in result.missing_keys:
        if key.startswith("enc_q") and for_infer:
            continue
        logger.warning(f"Missing key: {key}")
    for key in result.unexpected_keys:
        if key == "iteration":
            continue
        logger.warning(f"Unexpected key: {key}")
    if iteration is None:
        logger.info(f"Loaded '{checkpoint_path}'")
    else:
        logger.info(f"Loaded '{checkpoint_path}' (iteration {iteration})")

    return model, iteration


def save_safetensors(
    model: torch.nn.Module,
    iteration: int,
    checkpoint_path: Union[str, Path],
    is_half: bool = False,
    for_infer: bool = False,
) -> None:
    """
    モデルを safetensors 形式で保存する。

    Args:
        model (torch.nn.Module): 保存するモデル
        iteration (int): イテレーション回数
        checkpoint_path (Union[str, Path]): 保存先のパス
        is_half (bool): モデルを半精度で保存するかどうかのフラグ
        for_infer (bool): 推論用に読み込むかどうかのフラグ
    """

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    keys = []
    for k in state_dict:
        if "enc_q" in k and for_infer:
            continue
        keys.append(k)

    new_dict = (
        {k: state_dict[k].half() for k in keys}
        if is_half
        else {k: state_dict[k] for k in keys}
    )
    new_dict["iteration"] = torch.LongTensor([iteration])
    logger.info(f"Saved safetensors to {checkpoint_path}")

    save_file(new_dict, checkpoint_path)
