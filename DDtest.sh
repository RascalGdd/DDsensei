#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=check3KVD
#SBATCH --output=check3KVD.out
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
CUDA_VISIBLE_DEVICES=0 python test_kvd.py  --name check  --dataset_mode gtavtocityscapes --gpu_ids 0 \
--dataroot /data/public/cityscapes  \
--batch_size 2 --model_supervision 0  \
--netG 9 --channels_G 16 --kvd --no_EMA
