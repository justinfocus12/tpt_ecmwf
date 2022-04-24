#!/bin/bash
#SBATCH --job-name=request_DATASET
#SBATCH --output=request_DATASET.out
#SBATCH --error=request_DATASET.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB 
#SBATCH --time=1-12:00:00
python -u request_DATASET.py > request_DATASET_progress.txt
