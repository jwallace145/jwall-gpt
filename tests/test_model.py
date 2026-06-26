from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from jwall_gpt.data.dataset import MemmapDataset
from jwall_gpt.model.gpt import GPT
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
