from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from transformers import get_scheduler

from schatten_experiments.config import apply_training_overwrites, load_config, parse_key_value_items
from schatten_experiments.data import (
    build_accelerator,
    build_dataloaders,
    ensure_output_dir,
    load_model_and_tokenizer,
    load_raw_datasets,
    save_state_checkpoint,
    setup_logging,
)
from schatten_experiments.optimizers import Muon

logger = logging.getLogger(__name__)


def _split_muon_param_groups(model, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters into Muon and Adam-style groups."""
    muon_decay = []
    adam_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        low = name.lower()
        if "embed" in low or "lm_head" in low or param.ndim < 2:
            adam_no_decay.append(param)
        else:
            muon_decay.append(param)

    groups = []
    if muon_decay:
        groups.append({"params": muon_decay, "weight_decay": weight_decay, "muon": True})
    if adam_no_decay:
        groups.append({"params": adam_no_decay, "weight_decay": 0.0, "muon": False})
    return groups


def create_optimizer(model, config: dict[str, Any]):
    """Create a Muon optimizer; raises for unsupported optimizer names."""
    learning_rate = float(config["learning_rate"])
    momentum = float(config.get("momentum", 0.95))
    beta2 = float(config.get("beta2", 0.95))
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer_name = str(config.get("optimizer", "Muon"))

    if optimizer_name != "Muon":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Only 'Muon' is supported.")

    return Muon(
        _split_muon_param_groups(model, weight_decay),
        lr=learning_rate,
        momentum=momentum,
        betas=(momentum, beta2),
        weight_decay=weight_decay,
        adjust_lr_fn="match_rms_adamw",
        schatten_p=config.get("schatten_r", "inf"),
    )


def max_train_steps(config: dict[str, Any], train_dataloader) -> int:
    """Compute the total number of optimizer steps to run."""
    configured = config.get("max_train_steps")
    if configured is not None:
        return int(configured)
    updates_per_epoch = math.ceil(len(train_dataloader) / int(config.get("gradient_accumulation_steps", 1)))
    return int(config.get("num_train_epochs", 1)) * updates_per_epoch


def create_scheduler(config: dict[str, Any], optimizer, total_steps: int):
    """Create a learning rate scheduler from the config."""
    warmup_steps = int(config.get("num_warmup_steps", 0) or 0)
    warmup_ratio = config.get("warmup_ratio")
    if warmup_ratio is not None and float(warmup_ratio) > 0:
        warmup_steps = int(total_steps * float(warmup_ratio))

    scheduler_type = str(config.get("lr_scheduler_type", "linear"))
    scheduler_specific_kwargs = {}
    if scheduler_type == "warmup_stable_decay":
        decay_steps = int(total_steps * float(config.get("cooldown_proportion", 0.0) or 0.0))
        stable_steps = max(0, total_steps - warmup_steps - decay_steps)
        scheduler_specific_kwargs = {
            "num_decay_steps": decay_steps,
            "num_stable_steps": stable_steps,
        }

    return get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )


def current_learning_rates(scheduler, optimizer) -> list[float]:
    """Return the scheduler-visible learning rates for logging."""
    if hasattr(scheduler, "get_last_lr"):
        return [float(value) for value in scheduler.get_last_lr()]
    return [float(group["lr"]) for group in optimizer.param_groups]


@torch.no_grad()
def evaluate(accelerator, model, eval_dataloader, eval_batch_size: int) -> tuple[float, float]:
    """Evaluate the model and return loss/perplexity."""
    model.eval()
    losses = []
    progress = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process, desc="Evaluating")
    for _, batch in enumerate(eval_dataloader):
        outputs = model(**batch)
        losses.append(accelerator.gather_for_metrics(outputs.loss.repeat(eval_batch_size)))
        progress.update(1)

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = math.inf

    return eval_loss, perplexity


def save_effective_config(config: dict[str, Any], output_dir: str | Path) -> None:
    """Persist the resolved experiment config beside the run outputs."""
    output_path = Path(output_dir)
    with (output_path / "experiment_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with (output_path / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def run_training(config: dict[str, Any], output_dir: str | Path) -> dict[str, float]:
    """Run a full training loop and return evaluation metrics."""
    training = dict(config["training"])
    output_path = Path(output_dir)
    ensure_output_dir(output_path)

    accelerator = build_accelerator(training, output_path)
    setup_logging(accelerator, logger)
    if accelerator.is_main_process:
        save_effective_config(config, output_path)
    accelerator.wait_for_everyone()

    if training.get("with_tracking", False):
        accelerator.init_trackers(config.get("project_name", "schatten-train"), training)

    raw_datasets = load_raw_datasets(training)
    hf_config, model, tokenizer = load_model_and_tokenizer(training)
    train_dataloader, eval_dataloader, num_examples = build_dataloaders(
        raw_datasets,
        training,
        accelerator,
        hf_config,
        model,
        tokenizer,
        logger,
        train_batch_size=int(training["per_device_train_batch_size"]),
        eval_batch_size=int(training["per_device_eval_batch_size"]),
    )

    optimizer = create_optimizer(model, training)
    scheduler_steps = max_train_steps(training, train_dataloader)
    if training.get("max_train_steps") is not None:
        scheduler_steps *= accelerator.num_processes
    scheduler = create_scheduler(training, optimizer, scheduler_steps)

    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )
    total_steps = max_train_steps(training, train_dataloader)

    if accelerator.is_main_process:
        logger.info("Num examples = %s", num_examples)
        logger.info("Total optimization steps = %s", total_steps)
        logger.info("Scheduler steps = %s", scheduler_steps)

    if training.get("save_initialization", False):
        save_state_checkpoint(accelerator, model, tokenizer, output_path / "initialization")

    completed_steps = 0
    progress = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, desc="Training")

    num_epochs = int(training.get("num_train_epochs", 1))
    for epoch in range(num_epochs):
        model.train()
        log_loss = 0.0
        log_steps = 0
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1
                progress.update(1)
                log_loss += float(loss.detach().float().item())
                log_steps += 1

                if training.get("with_tracking", False) and training.get("tracking_log_steps", 0):
                    log_every = int(training["tracking_log_steps"])
                    if completed_steps % log_every == 0:
                        lrs = current_learning_rates(scheduler, optimizer)
                        payload = {
                            "train_loss_step": log_loss / max(1, log_steps),
                            "lr": lrs[0],
                            "stepsize": lrs[0],
                        }
                        payload.update({f"lr_group_{idx}": value for idx, value in enumerate(lrs)})
                        accelerator.log(payload, step=completed_steps)
                        log_loss = 0.0
                        log_steps = 0

                if completed_steps >= total_steps:
                    break
        if completed_steps >= total_steps:
            break

    last_eval_loss, last_perplexity = evaluate(
        accelerator,
        model,
        eval_dataloader,
        int(training["per_device_eval_batch_size"]),
    )
    if training.get("save_final_model", True):
        save_state_checkpoint(accelerator, model, tokenizer, output_path)

    results = {"eval_loss": last_eval_loss, "perplexity": last_perplexity}
    if accelerator.is_main_process:
        with (output_path / "all_results.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle)
    if training.get("with_tracking", False):
        accelerator.end_training()
    return results


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and launch a single training run."""
    parser = argparse.ArgumentParser(description="Train one Schatten Muon language-model run.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--overwrite", nargs="*", default=[], help="Training overwrites as KEY=VALUE.")
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    overwrites = parse_key_value_items(args.overwrite)
    config = apply_training_overwrites(config, overwrites)
    run_training(config, args.output_dir)


if __name__ == "__main__":
    main()
