from jwall_gpt.utils.config import GPTConfig

config = GPTConfig(
    n_layer=6,
    n_head=6,
    n_embd=384,
    block_size=256,
    vocab_size=50257,
    dropout=0.1,
)

batch_size = 32
max_steps = 500
eval_interval = 100
learning_rate = 3e-4
warmup_steps = 50
min_lr = 3e-5
grad_clip = 1.0
weight_decay = 0.1
checkpoint_dir = "checkpoints"
data_path = "data/train.bin"
