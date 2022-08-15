#!/bin/bash
#SBATCH --job-name=era5
#SBATCH --output=era5.out
#SBATCH --error=era5.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5GB 
#SBATCH --time=5:00:00
singularity exec --overlay /scratch/jf4241/software/overlay-7.5GB-300K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; conda activate base; python -u era5_request.py > progress_download-era5.txt"
