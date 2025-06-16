#!/bin/bash
#SBATCH --job-name=dml_full
#SBATCH --output=dml_output.log
#SBATCH --error=dml_error.log
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Gen

python DML_ATE.py


