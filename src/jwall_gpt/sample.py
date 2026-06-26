from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from jwall_gpt.model.gpt import GPT
from jwall_gpt.utils.checkpoint import load_checkpoint
from jwall_gpt.utils.config import GPTConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from a trained jwall-gpt checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]
    model_config = GPTConfig(**config_dict)

    device = get_device()
    model = GPT(model_config).to(device)
    load_checkpoint(args.checkpoint, model=model)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode(args.prompt)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = enc.decode(output_ids[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
