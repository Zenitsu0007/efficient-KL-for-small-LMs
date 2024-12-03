# Efficient Knowledge Distillation for Small Language Models

This repository is a replication project of the paper **Impossible Distillation: from Low-Quality Model to High-Quality Dataset & Model for Summarization and Paraphrasing**, with the addition of data evaluation pipeline and integration with Low-Rank Adaptation (LoRA) into the fine-tuning process. 

## Data Evaluation Setup

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

## Run Generation

```bash
sbatch run_generate_conditional_news.sh
```

When generating conditional, process one chunk of shards at a time by modifying the `--shard_start` and `--shard_size` arguments.

Example:

- Instance 1: --shard_start 0 --shard_size 1500
- Instance 2: --shard_start 1500 --shard_size 1500
- Instance 3: --shard_start 3000 --shard_size 1500

Set the `--part_idx` argument to the index of the chunk (0 to 5).

## LoRA Setup

```bash
!pip install pytorch_lightning
!pip install transformers
!pip install datasets
!pip install nltk
!pip install rouge_score
```

## Download datasets

```bash
!mkdir data
!wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz -P data
!tar -xvf data/paws_wiki_labeled_final.tar.gz -C data
!mv data/final/* data
!rm -r data/final
!rm -r data/paws_wiki_labeled_final.tar.gz
```

## Run

```bash
python T5/T5_finetune.py
```
