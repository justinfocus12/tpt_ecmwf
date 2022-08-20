#!/bin/bash
#SBATCH --job-name=s2s_request
#SBATCH --output=s2s_request_.out
#SBATCH --error=s2s_request.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB 
#SBATCH --time=05:00:00
singularity exec --overlay /scratch/jf4241/software/overlay-7.5GB-300K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; conda activate base; python -u s2s_request.py > progress_download-s2s.txt"
