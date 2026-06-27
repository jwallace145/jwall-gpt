from jwall_gpt.utils.config import GPTConfig

config = GPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    vocab_size=50257,
    dropout=0.1,
)

batch_size = 8
max_steps = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 6e-4
warmup_steps = 500
min_lr = 6e-5
grad_clip = 1.0
weight_decay = 0.1
checkpoint_dir = "checkpoints"
data_path = "data/train.bin"
val_data_path = "data/val.bin"
