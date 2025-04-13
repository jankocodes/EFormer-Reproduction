#!/bin/bash -l
#
#SBATCH --job-name=eformer_train        # Job name shown in squeue
#SBATCH --output=$WORK/EFormer-Reproduction/experiments/logs/%j.out
#SBATCH --error=$WORK/EFormer-Reproduction/experiments/logs/%j.err
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=24:00:00                 # Maximum runtime
#SBATCH --export=NONE                   # Do not inherit environment from submitting shell

unset SLURM_EXPORT_ENV                  # Required for srun to work correctly

# Load necessary modules
module load python/3.10-anaconda
module load cuda/11.8.0

# Activate Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate eformer_env

# Stage dataset to fast temporary storage and unpack 
echo "Copying dataset to FASTTMP..."
cd $TMPDIR
tar xf $WORK/dataset.tar 

# Change to project directory
cd $WORK/EFormer-Reproduction

# Run training script with staged data
srun python src/train.py

