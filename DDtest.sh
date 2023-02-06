#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=test
#SBATCH --output=test_c2c.lse.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --qos=batch


# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
source /usrhomes/s1422/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run your python code
# For single GPU use this
CUDA_VISIBLE_DEVICES=0 python test_no_generation.py
