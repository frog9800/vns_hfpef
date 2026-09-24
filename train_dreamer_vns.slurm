#!/bin/bash
#SBATCH --job-name=vns_hfpef_dreamer
#SBATCH --partition=lake-gpu          # L40S was fastest in the DropletRunner benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# Stage 3 only. Episodes are collected on CPU (train_offline_vns.py
# --stage collect) and scp'd here; the checkpoint goes back the same way.
# Training is fully offline, so there is no GPU-to-plant synchronization.

module purge
module load anaconda3
source activate dreamerv3

export DREAMERV3_PATH=$HOME/cyberrunner/dreamerv3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd $SLURM_SUBMIT_DIR

python -u train_offline_vns.py \
    --stage train \
    --episodes_dir episodes \
    --logdir logdir \
    --policy_dir policy \
    --steps 50000 \
    --batch_size 16 \
    --batch_length 64 \
    --decode_aux

# From the DropletRunner ablation: 50k-150k gradient steps on 50-100
# episodes was the useful range. 450k steps on the same data made every
# geometry worse, the world model memorizing episodes instead of
# generalizing. Add episodes before adding gradient steps.
