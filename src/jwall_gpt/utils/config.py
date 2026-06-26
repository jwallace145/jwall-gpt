from dataclasses import dataclass


@dataclass
class GPTConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    vocab_size: int = 50257
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            msg = f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            raise ValueError(msg)
