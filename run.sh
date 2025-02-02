# torchrun --standalone --nproc_per_node=8 train_gpt.py
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=45600 uv run train_gpt.py
