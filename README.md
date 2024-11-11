# Impossible-Distillation

This repository is a replication project of the paper **Impossible Distillation: from Low-Quality Model to High-Quality Dataset & Model for Summarization and Paraphrasing**.

# Setup

First install miniconda3.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

Then create a conda environment and install the dependencies.

```bash
conda create -n impossible-distillation python=3.12
conda activate impossible-distillation
conda install pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

# Run Generation

```bash
sbatch run_generate_conditional_news.sh
```

When generating conditional, process one chunk of shards at a time by modifying the `--shard_start` and `--shard_size` arguments.

Example:

- Instance 1: --shard_start 0 --shard_size 5000
- Instance 2: --shard_start 5000 --shard_size 5000
- Instance 3: --shard_start 10000 --shard_size 5000

Set the `--part_idx` argument to the index of the chunk (0 to 5).
