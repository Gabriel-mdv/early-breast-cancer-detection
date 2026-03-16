#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/jgibson2/projects/MobileFCMVitv3/slurm_logs/%x_%j.out
#SBATCH --error=/home/jgibson2/projects/MobileFCMVitv3/slurm_logs/%x_%j.err
#SBATCH --signal=B:SIGTERM@120

echo "==== MobileFCMViTv3 Evaluation ===="
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

cd /home/jgibson2/projects/MobileFCMVitv3
source venv/bin/activate

PROJECT_DIR=/home/jgibson2/projects/MobileFCMVitv3
mkdir -p $PROJECT_DIR/slurm_logs
mkdir -p $PROJECT_DIR/results

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running evaluation script..."
PYTHONPATH=$PROJECT_DIR python scripts/evaluate.py \
    --config_dir config/ \
    --checkpoint training/checkpoints/best_model.pt \
    --save_dir results

echo "Job finished at $(date)"
