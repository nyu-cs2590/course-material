#!/bin/bash
#SBATCH --job-name=job_wgpu
#SBATCH --account=csci_ga_2590-2023sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --bind /scratch --nv --overlay /scratch/nhj4247/overlay-25GB-500K.ext3:ro /scratch/nhj4247/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python ./test_gpu.py
"
