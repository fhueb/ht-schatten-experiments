"""Compute componentwise minibatch/full-gradient statistics for saved runs."""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from tqdm.auto import tqdm

from schatten_experiments.config import load_config
from schatten_experiments.data import (
    build_accelerator,
    build_dataloaders,
    checkpoint_model_dir,
    checkpoint_stats_dir,
    load_model_and_tokenizer,
    load_raw_datasets,
    setup_logging,
)
from schatten_experiments.stats.norms import (
    get_minibatch_stats,
    init_series_stats,
    resolve_norm_ps,
    summarize_tensor_dict,
)

logger = logging.getLogger(__name__)


@contextmanager
def disable_dropout(model):
    """Temporarily switch dropout modules to evaluation mode."""
    dropout_classes = (
        torch.nn.Dropout,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )
    modules = [module for module in model.modules() if isinstance(module, dropout_classes)]
    original = [module.training for module in modules]
    for module in modules:
        module.train(False)
    try:
        yield
    finally:
        for module, state in zip(modules, original):
            module.train(state)


@contextmanager
def full_gradient(
    accelerator,
    model,
    dataloader,
    *,
    save_path: Path | None = None,
) -> Iterator[tuple[dict[int, torch.Tensor], float]]:
    """Yield the full-dataset gradient and loss, optionally loading/saving a cache."""
    cached_grad = None
    cached_loss = None
    if save_path is not None and save_path.is_file():
        loaded = torch.load(save_path, map_location=accelerator.device)
        loaded_grad = loaded.get("grad", loaded) if isinstance(loaded, dict) else loaded
        cached_grad = {
            id(param): loaded_grad[name]
            for name, param in model.named_parameters()
            if name in loaded_grad
        }
        cached_loss = float(loaded.get("full_train_loss", 0.0)) if isinstance(loaded, dict) else 0.0
    if cached_grad is not None:
        yield cached_grad, cached_loss
        return

    model.train()
    num_batches = len(dataloader)
    grad_accum = getattr(accelerator, "gradient_accumulation_steps", 1)
    loss_sum = torch.tensor(0.0, device=accelerator.device)
    progress = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc="Full gradient")

    with disable_dropout(accelerator.unwrap_model(model)):
        for param in model.parameters():
            param.grad = None
        for batch in progress:
            with accelerator.no_sync(model):
                outputs = model(**batch)
                loss_sum += outputs.loss.detach().float()
                accelerator.backward(outputs.loss / max(1, num_batches) * grad_accum)

        accelerator.wait_for_everyone()
        for param in accelerator.unwrap_model(model).parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

        grad = {id(param): param.grad.detach().clone() for param in model.parameters() if param.grad is not None}
        full_train_loss = accelerator.reduce(loss_sum / max(1, num_batches), reduction="mean").item()
        for param in model.parameters():
            param.grad = None

    if save_path is not None and accelerator.is_main_process:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        by_name = {name: grad[id(param)].cpu() for name, param in model.named_parameters() if id(param) in grad}
        torch.save({"grad": by_name, "full_train_loss": full_train_loss}, save_path)

    yield grad, full_train_loss


def compute_noise_stats(config: dict, run_dir: str | Path, checkpoint: str) -> dict:
    """Compute and write noise statistics for one checkpoint of a training run."""
    data_config = dict(config["data"])
    examiner = dict(config["examiner"])
    run_path = Path(run_dir)
    model_dir = checkpoint_model_dir(run_path, checkpoint)
    stats_dir = checkpoint_stats_dir(run_path, checkpoint)

    run_config_json = run_path / "config.json"
    model_config_json = model_dir / "config.json"
    if run_config_json.is_file():
        data_config["config_name"] = str(run_config_json)
    elif model_config_json.is_file():
        data_config["config_name"] = str(model_config_json)
    else:
        data_config["config_name"] = str(model_dir)
    data_config["model_name_or_path"] = str(model_dir)

    accelerator = build_accelerator(
        {
            "gradient_accumulation_steps": data_config.get("gradient_accumulation_steps", 1),
            "mixed_precision": examiner.get("mixed_precision", "no"),
            "with_tracking": False,
            "seed": data_config.get("seed"),
        },
        stats_dir,
    )
    setup_logging(accelerator, logger)

    raw_datasets = load_raw_datasets(data_config)
    hf_config, model, tokenizer = load_model_and_tokenizer(data_config, checkpoint_dir=model_dir)
    dataloader, _, _ = build_dataloaders(
        raw_datasets,
        data_config,
        accelerator,
        hf_config,
        model,
        tokenizer,
        logger,
        train_batch_size=int(data_config["per_device_batch_size"]),
        eval_batch_size=int(data_config["per_device_batch_size"]),
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    norm_ps = resolve_norm_ps(examiner.get("norm_ps"), with_schatten=bool(examiner.get("with_schatten", False)))
    param_items = list(model.named_parameters())
    param_id_to_name = {id(param): name for name, param in param_items}
    param_name_to_ndim = {name: param.ndim for name, param in param_items}

    full_grad_path = model_dir / "full_grad.pt" if examiner.get("save_full_grad", False) else None
    with full_gradient(accelerator, model, dataloader, save_path=full_grad_path) as (full_grad, full_train_loss):
        stats = init_series_stats(
            param_name_to_ndim,
            norm_ps,
            with_schatten=bool(examiner.get("with_schatten", False)),
        )
        full_summary, full_componentwise, full_singular_values = summarize_tensor_dict(
            full_grad,
            norm_ps,
            param_id_to_name,
            with_schatten=bool(examiner.get("with_schatten", False)),
            include_singular_values=True,
        )
        stats["full_norm"] = full_summary
        stats["componentwise_full_norm"] = full_componentwise
        stats["full_singular_values_2d"] = full_singular_values
        stats["full_train_loss"] = full_train_loss

        progress = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc="Noise minibatches")
        for batch in progress:
            model.train()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                accelerator.backward(outputs.loss)
            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    batch_grad = {
                        id(param): param.grad.detach().clone()
                        for param in model.parameters()
                        if param.grad is not None
                    }
                    get_minibatch_stats(
                        batch_grad,
                        full_grad,
                        norm_ps,
                        stats,
                        param_id_to_name,
                        with_schatten=bool(examiner.get("with_schatten", False)),
                    )
                model.zero_grad()

    entry = {
        "model_name_or_path": str(model_dir),
        "resume_from_checkpoint": str(model_dir),
        "stats": stats,
    }
    if accelerator.is_main_process:
        stats_dir.mkdir(parents=True, exist_ok=True)
        with (stats_dir / "opt_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(entry, handle)
    accelerator.free_memory()
    return entry


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and process the requested checkpoint set."""
    parser = argparse.ArgumentParser(description="Compute noise statistics for a completed Schatten run.")
    parser.add_argument("--config", required=True, help="YAML noise config.")
    parser.add_argument("--run-dir", required=True, help="Completed training run directory.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    for checkpoint in config["examiner"]["checkpoints"]:
        compute_noise_stats(config, args.run_dir, checkpoint)


if __name__ == "__main__":
    main()
