from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jwall_gpt.data.dataset import MemmapDataset
from jwall_gpt.model.gpt import GPT
from jwall_gpt.utils.checkpoint import load_checkpoint, save_checkpoint
from jwall_gpt.utils.config import GPTConfig

PROVENANCE_ENV = {
    "release_tag": "RELEASE_TAG",
    "dataset": "DATASET",
    "tokenizer": "TOKENIZER",
    "run_id": "RUN_ID",
}


def summarize_metrics(history: list[dict]) -> dict:
    """Reduce per-eval records into summary stats for ``run.json``.

    ``history`` is a list of ``{"step", "train_loss", "val_loss", "lr"}`` dicts.
    ``val_loss`` may be ``None`` when no validation set is configured.
    """
    train_points = [h for h in history if h.get("train_loss") is not None]
    val_points = [h for h in history if h.get("val_loss") is not None]
    best = min(val_points, key=lambda h: h["val_loss"]) if val_points else None
    return {
        "final_train_loss": train_points[-1]["train_loss"] if train_points else None,
        "final_val_loss": val_points[-1]["val_loss"] if val_points else None,
        "best_val_loss": best["val_loss"] if best else None,
        "best_val_step": best["step"] if best else None,
    }


def collect_provenance() -> dict:
    """Read run provenance (release tag, dataset, run id, ...) from the environment."""
    return {field: os.environ.get(var) for field, var in PROVENANCE_ENV.items()}


def load_config_module(config_path: Path):
    spec = importlib.util.spec_from_file_location("train_config", config_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load config from {config_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_config"] = module
    spec.loader.exec_module(module)
    return module


def get_lr(
    step: int, *, warmup_steps: int, learning_rate: float, min_lr: float, max_steps: int
) -> float:
    if step < warmup_steps:
        return learning_rate * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(model: GPT, loader: DataLoader, device: torch.device, eval_iters: int) -> float:
    """Average the loss over up to ``eval_iters`` batches with the model in eval mode.

    Restores training mode before returning so the caller can resume optimization.
    """
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= eval_iters:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train jwall-gpt")
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.py"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps from the config (useful for longer cloud runs)",
    )
    args = parser.parse_args()

    cfg = load_config_module(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    model_config: GPTConfig = cfg.config
    device = get_device()

    data_path = Path(cfg.data_path)
    if not data_path.exists():
        msg = f"Data file not found: {data_path}. Run scripts/preprocess.py first."
        raise FileNotFoundError(msg)

    dataset = MemmapDataset(data_path, model_config.block_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    val_loader: DataLoader | None = None
    val_data_path_str = getattr(cfg, "val_data_path", None)
    if val_data_path_str:
        val_data_path = Path(val_data_path_str)
        if val_data_path.exists():
            val_dataset = MemmapDataset(val_data_path, model_config.block_size)
            val_loader = DataLoader(
                val_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
            )
        else:
            print(f"No validation data at {val_data_path}; skipping eval loss.")
    eval_iters = getattr(cfg, "eval_iters", 50)

    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    start_step = 0
    checkpoint_dir = Path(cfg.checkpoint_dir)
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, model=model, optimizer=optimizer)
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    print(f"Device: {device}")
    print(f"Parameters: {model.count_parameters():,}")

    started_at = datetime.now(UTC)
    start_perf = time.perf_counter()
    history: list[dict] = []
    run_json_path = checkpoint_dir / "run.json"
    train_settings = {
        "batch_size": cfg.batch_size,
        "max_steps": cfg.max_steps,
        "learning_rate": cfg.learning_rate,
        "warmup_steps": cfg.warmup_steps,
        "min_lr": cfg.min_lr,
        "grad_clip": cfg.grad_clip,
        "weight_decay": cfg.weight_decay,
        "eval_interval": cfg.eval_interval,
        "eval_iters": eval_iters,
    }

    def write_run_metadata(final_step: int) -> None:
        metadata = {
            "config_path": str(args.config),
            "config_name": args.config.stem,
            "device": str(device),
            "parameters": model.count_parameters(),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "duration_seconds": round(time.perf_counter() - start_perf, 2),
            "provenance": collect_provenance(),
            "model_config": dict(model_config.__dict__),
            "train_config": train_settings,
            "metrics": {
                "start_step": start_step,
                "final_step": final_step,
                "total_steps": final_step - start_step + 1,
                **summarize_metrics(history),
                "history": history,
            },
        }
        run_json_path.write_text(json.dumps(metadata, indent=2))

    model.train()
    step = start_step
    while step < cfg.max_steps:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            lr = get_lr(
                step,
                warmup_steps=cfg.warmup_steps,
                learning_rate=cfg.learning_rate,
                min_lr=cfg.min_lr,
                max_steps=cfg.max_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if step % cfg.eval_interval == 0:
                train_loss = loss.item()
                val_loss = (
                    estimate_loss(model, val_loader, device, eval_iters)
                    if val_loader is not None
                    else None
                )
                history.append(
                    {"step": step, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
                )
                log = f"step {step}: train loss {train_loss:.4f}"
                if val_loss is not None:
                    log += f", val loss {val_loss:.4f}"
                log += f", lr {lr:.2e}"
                print(log)
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    config=model_config.__dict__,
                )
                write_run_metadata(step)

            step += 1
            if step >= cfg.max_steps:
                break

    final_step = step - 1
    save_checkpoint(
        checkpoint_dir / "latest.pt",
        step=final_step,
        model=model,
        optimizer=optimizer,
        config=model_config.__dict__,
    )
    write_run_metadata(final_step)
    print(f"Training complete. Checkpoint saved to {checkpoint_dir / 'latest.pt'}")
    print(f"Run metadata saved to {run_json_path}")


if __name__ == "__main__":
    main()
