from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from jwall_gpt.data.dataset import MemmapDataset
from jwall_gpt.data.prepare import split_tokens
from jwall_gpt.model.gpt import GPT
from jwall_gpt.train import collect_provenance, estimate_loss, summarize_metrics
from jwall_gpt.utils.checkpoint import load_checkpoint, save_checkpoint
from jwall_gpt.utils.config import GPTConfig


@pytest.fixture
def tiny_config() -> GPTConfig:
    return GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
        vocab_size=100,
        dropout=0.0,
    )


def test_gpt_forward_pass(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config)
    batch, seq = 2, 8
    idx = torch.randint(0, tiny_config.vocab_size, (batch, seq))
    targets = torch.randint(0, tiny_config.vocab_size, (batch, seq))
    logits, loss = model(idx, targets)
    assert logits.shape == (batch, seq, tiny_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_gpt_forward_without_targets(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    logits, loss = model(idx)
    assert logits.shape == (1, 4, tiny_config.vocab_size)
    assert loss is None


def test_gpt_generate(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config)
    model.eval()
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 9)


def test_gpt_count_parameters(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config)
    assert model.count_parameters() > 0


def test_gpt_rejects_long_sequence(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size + 1))
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(idx)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(n_embd=10, n_head=3)


def test_memmap_dataset() -> None:
    tokens = np.arange(100, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.bin"
        tokens.tofile(path)
        dataset = MemmapDataset(path, block_size=8)
        assert len(dataset) == 92
        x, y = dataset[0]
        assert x.shape == (8,)
        assert y.shape == (8,)
        assert y[0].item() == x[1].item()


def test_split_tokens_holds_out_tail() -> None:
    tokens = np.arange(100, dtype=np.uint16)
    train, val = split_tokens(tokens, val_frac=0.1)
    assert len(train) == 90
    assert len(val) == 10
    assert train[0] == 0
    assert val[0] == 90
    assert val.dtype == tokens.dtype


def test_split_tokens_zero_frac_yields_empty_val() -> None:
    tokens = np.arange(50, dtype=np.uint16)
    train, val = split_tokens(tokens, val_frac=0.0)
    assert len(train) == 50
    assert len(val) == 0
    assert val.dtype == tokens.dtype


@pytest.mark.parametrize("val_frac", [-0.1, 1.0, 1.5])
def test_split_tokens_rejects_invalid_frac(val_frac: float) -> None:
    tokens = np.arange(10, dtype=np.uint16)
    with pytest.raises(ValueError, match="val_frac"):
        split_tokens(tokens, val_frac=val_frac)


def test_estimate_loss_returns_mean_and_restores_train_mode(tiny_config: GPTConfig) -> None:
    tokens = np.arange(200, dtype=np.uint16) % tiny_config.vocab_size
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "val.bin"
        tokens.astype(np.uint16).tofile(path)
        dataset = MemmapDataset(path, tiny_config.block_size)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)

        model = GPT(tiny_config)
        model.train()
        loss = estimate_loss(model, loader, torch.device("cpu"), eval_iters=3)

        assert isinstance(loss, float)
        assert loss > 0.0
        assert model.training is True


def test_summarize_metrics_with_val_losses() -> None:
    history = [
        {"step": 0, "train_loss": 4.0, "val_loss": 4.2, "lr": 1e-4},
        {"step": 100, "train_loss": 2.0, "val_loss": 2.1, "lr": 3e-4},
        {"step": 200, "train_loss": 1.5, "val_loss": 2.5, "lr": 2e-4},
    ]
    summary = summarize_metrics(history)
    assert summary["final_train_loss"] == 1.5
    assert summary["final_val_loss"] == 2.5
    assert summary["best_val_loss"] == 2.1
    assert summary["best_val_step"] == 100


def test_summarize_metrics_without_val_losses() -> None:
    history = [
        {"step": 0, "train_loss": 4.0, "val_loss": None, "lr": 1e-4},
        {"step": 100, "train_loss": 2.0, "val_loss": None, "lr": 3e-4},
    ]
    summary = summarize_metrics(history)
    assert summary["final_train_loss"] == 2.0
    assert summary["final_val_loss"] is None
    assert summary["best_val_loss"] is None
    assert summary["best_val_step"] is None


def test_summarize_metrics_empty_history() -> None:
    summary = summarize_metrics([])
    assert summary == {
        "final_train_loss": None,
        "final_val_loss": None,
        "best_val_loss": None,
        "best_val_step": None,
    }


def test_collect_provenance_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELEASE_TAG", "v1.2.3")
    monkeypatch.setenv("DATASET", "tinystories")
    monkeypatch.delenv("TOKENIZER", raising=False)
    monkeypatch.delenv("RUN_ID", raising=False)
    provenance = collect_provenance()
    assert provenance["release_tag"] == "v1.2.3"
    assert provenance["dataset"] == "tinystories"
    assert provenance["tokenizer"] is None
    assert provenance["run_id"] is None


def test_checkpoint_roundtrip(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            path,
            step=10,
            model=model,
            optimizer=optimizer,
            config=tiny_config.__dict__,
        )
        model2 = GPT(tiny_config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        ckpt = load_checkpoint(path, model=model2, optimizer=optimizer2)
        assert ckpt["step"] == 10
        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.allclose(p1, p2)
