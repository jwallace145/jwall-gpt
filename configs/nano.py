from jwall_gpt.utils.config import GPTConfig

config = GPTConfig(
    n_layer=8,
    n_head=8,
    n_embd=512,
    block_size=512,
    vocab_size=50257,
    dropout=0.1,
)

batch_size = 16
max_steps = 2000
eval_interval = 200
eval_iters = 100
learning_rate = 3e-4
warmup_steps = 200
min_lr = 3e-5
grad_clip = 1.0
weight_decay = 0.1
checkpoint_dir = "checkpoints"
data_path = "data/train.bin"
val_data_path = "data/val.bin"
