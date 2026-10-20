#!/bin/bash
# Batch-transfer cross-validation on one GPU. Arguments are passed to the script as Hydra overrides.
#
#   sbatch -J controls_split hpc/train_batch_transfer.sbatch name=controls_split
#   sbatch -J p32 hpc/train_batch_transfer.sbatch name=p32 neural.n_features=32
#   sbatch -p gpu -w gpu-350-04 -J a4500 hpc/train_batch_transfer.sbatch name=a4500   # pick a node
#
# Submit from the repo root. Log: outputs/slurm-<job name>-<job id>.out; results go to the Hydra
# run dir outputs/<date>/<name>_<time>/. Flags on the sbatch command line override those below.
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# what was run: commit, uncommitted changes, GPU
echo "job $SLURM_JOB_ID on $(hostname): $*"
git rev-parse --short HEAD && git status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run python scripts/train_batch_transfer.py "$@"
