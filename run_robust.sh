#!/bin/bash
#SBATCH --job-name=robust_dml
#SBATCH --output=robust_model.log
#SBATCH --error=robust_model_err.log
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Gen

python DML_robust_model.py
