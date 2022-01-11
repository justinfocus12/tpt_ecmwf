#!/bin/bash
#SBATCH --job-name=s2s
#SBATCH --output=s2s.out
#SBATCH --error=s2s.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15GB 
#SBATCH --time=00:20:00
singularity exec --overlay /scratch/jf4241/software/overlay-7.5GB-300K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python driver2.py"
