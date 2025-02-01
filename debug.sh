# uv run torchrun --standalone --no-python --nproc_per_node=1 ipython --pdb train_gpt.py
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=45600 uv run ipython --pdb train_gpt.py
