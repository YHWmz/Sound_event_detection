#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
# CUBLAS_WORKSPACE_CONFIG=:4096:8 
python run.py train_evaluate configs/baseline.yaml data/eval/feature.csv data/eval/label.csv

