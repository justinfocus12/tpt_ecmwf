#!/bin/bash
#SBATCH --job-name=s2s_request
#SBATCH --output=s2s_request.out
#SBATCH --error=s2s_request.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB 
#SBATCH --time=1-12:00:00
python -u s2s_request.py > s2s_request.txt
