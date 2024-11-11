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
sbatch run_generation_news.sh
```
