#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=3
#SBATCH --mem=92G
#SBATCH --time=4:00:00
#SBATCH --qos=normal
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


# ---- Run training ----
CONFIG_DIR=${1:-config/}
RUN_NAME=${2:-baseline}
echo "Running training: config=$CONFIG_DIR run=$RUN_NAME"
PYTHONPATH=$PROJECT_DIR python scripts/train.py --config_dir $CONFIG_DIR --run_name $RUN_NAME

echo "Job finished at $(date)"
