#!/bin/bash
#SBATCH --job-name=dml_cate
#SBATCH --output=dml_cate.log
#SBATCH --error=dml_cate_err.log
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Gen

python DML_CATE.py
