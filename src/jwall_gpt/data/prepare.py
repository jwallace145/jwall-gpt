from __future__ import annotations

import numpy as np
import tiktoken


def tokenize_corpus(text: str) -> np.ndarray:
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    return np.array(tokens, dtype=np.uint16)


def split_tokens(tokens: np.ndarray, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a token stream into contiguous train/val arrays.

    The final ``val_frac`` of the stream becomes the validation set so the two
    splits never overlap. ``val_frac`` of 0 yields an empty validation array.
    """
    if not 0.0 <= val_frac < 1.0:
        msg = f"val_frac must be in [0.0, 1.0), got {val_frac}"
        raise ValueError(msg)

    n_val = int(len(tokens) * val_frac)
    if n_val == 0:
        return tokens, np.empty(0, dtype=tokens.dtype)

    split = len(tokens) - n_val
    return tokens[:split], tokens[split:]
