#!/bin/bash

#SBATCH --job-name=svm
#SBATCH --output=%j_svm.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --partition=general

source activate /data/group_data/neuroagents_lab/conda_envs/tactile
python /home/trinityc/tactile/tactile-nn/tactile_model/svm_crossval.py \
  +data_dir="/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed" \
  +split="validate"