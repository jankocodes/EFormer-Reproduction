#!/bin/bash -l
#
#SBATCH --job-name=eformer_train        # Job name shown in squeue
#SBATCH --output=/home/woody/mrrr/mrrr107v/ablations/bilinear_only/%j.out
#SBATCH --error=/home/woody/mrrr/mrrr107v/ablations/bilinear_only/%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00                 # Maximum runtime
#SBATCH --export=NONE                   # Do not inherit environment from submitting shell

unset SLURM_EXPORT_ENV

echo "[$(date)] Job started on node: $(hostname)"
echo "[$(date)] SLURM job ID: $SLURM_JOB_ID"
echo "[$(date)] Checking environment setup..."

# Load modules
echo "Loading Python and CUDA modules..."
module load python/3.10-anaconda || { echo "Failed to load python module"; exit 1; }
module load cuda/11.8.0 || { echo "Failed to load cuda module"; exit 1; }

# Check conda
echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }
conda activate eformer_env || { echo "Failed to activate conda env"; exit 1; }

# Stage data
echo "[$(date)] Copying dataset to TMPDIR: $TMPDIR"
cd $TMPDIR || { echo "Failed to cd into TMPDIR"; exit 1; }
tar xf /home/woody/mrrr/mrrr107v/dataset.tar || { echo "Failed to unpack dataset.tar"; exit 1; }

# Check dataset
echo "Verifying extracted dataset..."
ls -l $TMPDIR
ls -l $TMPDIR/datasets/composite_dataset/train

# Project directory
echo "[$(date)] Changing to project dir: /home/woody/mrrr/mrrr107v/EFormer-Reproduction"
cd /home/woody/mrrr/mrrr107v/EFormer-Reproduction || { echo "Failed to cd into project directory"; exit 1; }

# Test script run
echo "[$(date)] Running a dry Python command to test setup..."
srun python -c "print('Python environment is working')" || { echo "Python test failed"; exit 1; }

echo "[$(date)] All checks passed. Ready to run training script."


# Run training script with staged data
srun python src/evaluate.py \
    --run_name $SLURM_JOB_ID \
    --model_path /home/woody/mrrr/mrrr107v/ablations/bilinear_transconv/checkpoints/best_model.pth \
    --first_upsampling bilinear \
    --second_upsampling transconv \
    --data_root $TMPDIR/datasets/composite_dataset \
    --out_dir /home/woody/mrrr/mrrr107v/ablations/bilinear_transconv