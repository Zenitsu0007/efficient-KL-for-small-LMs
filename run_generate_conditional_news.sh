#!/bin/bash
# The interpreter used to execute the script

# Lines beginning with #SBATCH specify your computing resources and other logistics about how to run your job.

#SBATCH --job-name=news_con_part_0_21630_to_22830
#SBATCH --account=eecs498s006f24_class
#SBATCH --partition=spgpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30g
#SBATCH --output=news_con_part_0_21630_to_22830.log

# Load Python and any other desired modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate impossible-distillation

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

# Specify the script you want to run
python generate_conditional.py --part_idx 0 --con_stage 0 --con_domain news --con_model_name gpt2-xl --device_id 0 --shard_size 1200 --shard_start 21630 --save_size 10
