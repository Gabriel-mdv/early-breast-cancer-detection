#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jgibson2/projects/MobileFCMVitv3/slurm_logs/%x_%j.out
#SBATCH --error=/home/jgibson2/projects/MobileFCMVitv3/slurm_logs/%x_%j.err
#SBATCH --signal=B:SIGTERM@120


echo "==== MobileFCMViTv3 Training ===="

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"


# ---- Load environment ----
cd /home/jgibson2/projects/MobileFCMVitv3
source venv/bin/activate


PROJECT_DIR=/home/jgibson2/projects/MobileFCMVitv3
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints
SLURM_LOG_DIR=$PROJECT_DIR/slurm_logs
mkdir -p $CHECKPOINT_DIR
mkdir -p $SLURM_LOG_DIR


export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0


# ---- Run training ----
echo "Running training script..."
PYTHONPATH=$PROJECT_DIR python scripts/train.py --config_dir config/

echo "Job finished at $(date)"
