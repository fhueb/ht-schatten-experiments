from __future__ import annotations

import datetime
import logging
import os
from itertools import chain
from pathlib import Path
from typing import Any

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
import datasets
from datasets import load_dataset
import transformers
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)


def build_accelerator(config: dict[str, Any], output_dir: str | Path | None = None) -> Accelerator:
    """Create an Accelerator with the long timeout needed by full-gradient passes."""
    kwargs = {}
    if config.get("with_tracking", False):
        report_to = config.get("report_to")
        if report_to:
            kwargs["log_with"] = report_to
        kwargs["project_dir"] = str(output_dir) if output_dir is not None else None
    process_group_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))
    mixed_precision = config.get("mixed_precision")
    if mixed_precision is None:
        mixed_precision = "no"
    mixed_precision = None if str(mixed_precision).lower() in {"", "none", "no"} else str(mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        mixed_precision=mixed_precision,
        kwargs_handlers=[process_group_kwargs],
        **kwargs,
    )
    seed = config.get("seed")
    if seed is not None:
        set_seed(int(seed))
    return accelerator


def setup_logging(accelerator: Accelerator, logger: logging.Logger) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("%s", accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


def _split_percentage_token(value: float) -> str:
    if value <= 0 or value >= 100:
        raise ValueError("validation_split_percentage must be in the open interval (0, 100).")
    return str(int(value)) if float(value).is_integer() else str(value)


def load_raw_datasets(config: dict[str, Any]):
    """Load the configured Hugging Face dataset with the configured split convention."""
    dataset_name = config.get("dataset_name")
    if dataset_name is None:
        raise ValueError("dataset_name is required.")
    dataset_config_name = config.get("dataset_config_name")
    dataset_revision = config.get("dataset_revision")
    trust_remote_code = bool(config.get("trust_remote_code", False))
    validation_split_percentage = float(config.get("validation_split_percentage", 5))
    validation_split = _split_percentage_token(validation_split_percentage) if validation_split_percentage > 0 else None

    raw = load_dataset(
        dataset_name,
        dataset_config_name,
        revision=dataset_revision,
        trust_remote_code=trust_remote_code,
    )
    if "validation" not in raw and validation_split is not None:
        raw["validation"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{validation_split}%]",
            revision=dataset_revision,
            trust_remote_code=trust_remote_code,
        )
        raw["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[{validation_split}%:]",
            revision=dataset_revision,
            trust_remote_code=trust_remote_code,
        )
    return raw


def load_model_and_tokenizer(config: dict[str, Any], *, checkpoint_dir: str | Path | None = None):
    """Load tokenizer/config and either instantiate or load a causal LM."""
    logger = logging.getLogger(__name__)
    trust_remote_code = bool(config.get("trust_remote_code", False))
    model_name_or_path = str(checkpoint_dir) if checkpoint_dir is not None else config.get("model_name_or_path")
    config_name = config.get("config_name") or model_name_or_path
    tokenizer_name = config.get("tokenizer_name") or model_name_or_path
    config_revision = config.get("config_revision")
    model_revision = config.get("model_revision")
    tokenizer_revision = config.get("tokenizer_revision")

    if not config_name:
        raise ValueError("config_name or model_name_or_path is required.")
    hf_config = AutoConfig.from_pretrained(config_name, revision=config_revision, trust_remote_code=trust_remote_code)

    if not model_name_or_path and hasattr(hf_config, "torch_dtype"):
        hf_config.torch_dtype = None
    if checkpoint_dir is not None and Path(checkpoint_dir).joinpath("config.json").is_file():
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            config=hf_config,
            revision=model_revision,
            trust_remote_code=trust_remote_code,
        )
    elif model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=hf_config,
            from_tf=bool(".ckpt" in str(model_name_or_path)),
            revision=model_revision,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=trust_remote_code)

    if tokenizer_name is None:
        raise ValueError("tokenizer_name is required.")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=not bool(config.get("use_slow_tokenizer", False)),
        revision=tokenizer_revision,
        trust_remote_code=trust_remote_code,
    )
    return hf_config, model, tokenizer


def _block_size(config: dict[str, Any], tokenizer, hf_config, logger: logging.Logger) -> int:
    block_size = config.get("block_size")
    if block_size is None:
        block_size = tokenizer.model_max_length
        max_positions = getattr(hf_config, "max_position_embeddings", block_size)
        if block_size > max_positions:
            block_size = min(1024, max_positions)
            logger.warning("Tokenizer max length is large; using block_size=%s.", block_size)
    else:
        block_size = min(int(block_size), tokenizer.model_max_length)
    return int(block_size)


def build_dataloaders(
    raw_datasets,
    config: dict[str, Any],
    accelerator: Accelerator,
    hf_config,
    model,
    tokenizer,
    logger: logging.Logger,
    *,
    train_batch_size: int,
    eval_batch_size: int,
):
    """Tokenize, block, optionally subsample, and return train/eval dataloaders."""
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names
    text_column = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column])

    with accelerator.main_process_first():
        tokenized = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=config.get("preprocessing_num_workers"),
            remove_columns=column_names,
            load_from_cache_file=not bool(config.get("overwrite_cache", False)),
            desc="Tokenizing dataset",
        )

    block_size = _block_size(config, tokenizer, hf_config, logger)

    def group_texts(examples):
        concatenated = {key: list(chain(*examples[key])) for key in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            key: [tokens[i : i + block_size] for i in range(0, total_length, block_size)]
            for key, tokens in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized.map(
            group_texts,
            batched=True,
            num_proc=config.get("preprocessing_num_workers"),
            load_from_cache_file=not bool(config.get("overwrite_cache", False)),
            desc=f"Grouping texts into {block_size}-token chunks",
        )

    train_dataset = lm_datasets["train"]
    if "validation" not in lm_datasets:
        raise ValueError("Validation dataset is missing. Set validation_split_percentage > 0.")
    eval_dataset = lm_datasets["validation"]
    proportion = config.get("proportion_of_dataset")
    if proportion is not None:
        train_count = max(1, int(len(train_dataset) * float(proportion)))
        eval_count = max(1, int(len(eval_dataset) * float(proportion)))
        train_dataset = train_dataset.select(range(train_count))
        eval_dataset = eval_dataset.select(range(eval_count))

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after preprocessing.")
    if len(eval_dataset) == 0:
        raise ValueError("Validation dataset is empty after preprocessing.")
    if train_batch_size > len(train_dataset):
        raise ValueError(
            "Training dataset has fewer preprocessed examples than per-device train batch size. "
            "Reduce per_device_train_batch_size, reduce block_size, or use more text."
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=int(train_batch_size),
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=int(eval_batch_size),
    )
    return train_dataloader, eval_dataloader, len(train_dataset)


def save_state_checkpoint(accelerator: Accelerator, model, tokenizer, output_dir: str | Path) -> None:
    """Save a model/tokenizer checkpoint in Hugging Face format for later inspection."""
    output_path = Path(output_dir)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(output_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_path)


def checkpoint_model_dir(run_dir: str | Path, checkpoint: str) -> Path:
    """Resolve a logical checkpoint name to a model directory."""
    run_path = Path(run_dir)
    if checkpoint in {"", "final"}:
        return run_path
    return run_path / checkpoint


def checkpoint_stats_dir(run_dir: str | Path, checkpoint: str) -> Path:
    name = "final" if checkpoint in {"", "final"} else checkpoint
    return Path(run_dir) / "stats" / name


def ensure_output_dir(path: str | Path | None) -> None:
    if path is not None:
        os.makedirs(path, exist_ok=True)
