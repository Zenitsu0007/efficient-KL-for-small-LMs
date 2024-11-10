#!/bin/bash
# The interpreter used to execute the script

# Lines beginning with #SBATCH specify your computing resources and other logistics about how to run your job.

#SBATCH --job-name=news_generation
#SBATCH --account=eecs498s006f24_class
#SBATCH --partition=spgpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30g
#SBATCH --output=news_generation.log

# Load Python and any other desired modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate impossible-distillation

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

# Specify the script you want to run
python generate_context.py --device_id 0 --domain news --seed 445 --total_size 500000
