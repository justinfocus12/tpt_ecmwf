#!/bin/bash
#SBATCH --job-name=s2s
#SBATCH --output=s2s.out
#SBATCH --error=s2s.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB 
#SBATCH --time=05:00:00
#conda activate /scratch/jf4241/penv
python -u driver2.py > s2s_msg.txt

