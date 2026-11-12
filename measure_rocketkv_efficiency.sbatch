#!/bin/bash
#SBATCH --job-name=measure_rocketkv_efficiency     # name for your job
#SBATCH --nodes=1                     # number of nodes
#SBATCH --ntasks-per-node=1           # number of processes per node
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8             # number of CPU cores           
#SBATCH --time=02:00:00               # total run time limit (HH:MM:SS)      
#SBATCH -o ./slurm_outs/slurm_%j.out  # file to write logs, prints, etc
#SBATCH --mail-type=begin             # send email when job begins
#SBATCH --mail-type=end               # send email when job ends

set -Eeuo pipefail

source .venv/bin/activate
uv pip install --no-build-isolation flash-attn==2.6.3

# First prepare llama3.1-8b for experiments
bash scripts/dataset_prep/download_gpt_fast.sh

# Get baseline e2e request latency and mem. usage metrics at different context lengths
bash scripts/efficiency/llama3.1-8b-instruct.sh rocket results 256
bash scripts/efficiency/llama3.1-8b-instruct.sh rocket results 512
bash scripts/efficiency/llama3.1-8b-instruct.sh rocket results 1024
bash scripts/efficiency/llama3.1-8b-instruct.sh rocket results 2048
bash scripts/efficiency/llama3.1-8b-instruct.sh rocket results 4096
