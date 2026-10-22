#!/bin/bash
# Run one script on one GPU. First argument: the script; the rest are its Hydra overrides.
#
#   sbatch -J split_all hpc/run.sbatch scripts/train_batch_transfer_expression.py name=split_all
#   sbatch -J p32 hpc/run.sbatch scripts/train_batch_transfer_expression.py name=p32 neural.n_features=32
#   sbatch -J tune --time=8:00:00 hpc/run.sbatch scripts/tune_bridges.py name=tune
#   sbatch -p gpu -w gpu-350-04 -J a4500 hpc/run.sbatch scripts/train_batch_transfer_expression.py name=a4500
#
# Submit from the repo root. Log: outputs/slurm-<job name>-<job id>.out; results go to the Hydra
# run dir outputs/<date>/<name>_<time>/. Flags on the sbatch command line override those below.
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"  # uv; batch jobs don't read ~/.bashrc
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TQDM_DISABLE=1  # progress bars would flood the log

# what was run: commit, uncommitted changes, GPU
echo "job $SLURM_JOB_ID on $(hostname): $*"
git rev-parse --short HEAD && git status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

script=$1
shift
uv run python "$script" "$@"
