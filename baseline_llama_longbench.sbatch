#!/bin/bash
#SBATCH --job-name=baseline_llama_longbench     # name for your job
#SBATCH --nodes=1                     # number of nodes
#SBATCH --ntasks-per-node=1           # number of processes per node
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8             # number of CPU cores           
#SBATCH --time=12:00:00               # total run time limit (HH:MM:SS)      
#SBATCH -o ./slurm_outs/slurm_%j.out  # file to write logs, prints, etc
#SBATCH --mail-type=begin             # send email when job begins
#SBATCH --mail-type=end               # send email when job ends

set -Eeuo pipefail

source .venv/bin/activate
uv pip install --no-build-isolation flash-attn==2.6.3

# First prepare llama3.1-8b for experiments
bash scripts/dataset_prep/download_gpt_fast.sh

# download longbench dataset
uv run python scripts/dataset_prep/download_longbench.py

# reproduce longbench accuracy vs. token budget curves
bash scripts/longbench/Llama-3.2-3B-Instruct-FP8.sh baseline results 
