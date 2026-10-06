#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# #SBATCH --output=/dev/null
# #SBATCH --error=/dev/null
# module load gcc/12.3.0
set -euo pipefail

# Submit with `sbatch train.sh` from the repository root.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${REPO_ROOT}/build/tagc.sif"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${SLURM_JOB_ID}}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29501}"
NUM_NODES="$SLURM_JOB_NUM_NODES"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

mkdir -p "$LOG_DIR"

# --cleanenv requires explicit forwarding of application environment variables.
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_DATASETS_OFFLINE=1
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
for name in WANDB_API_KEY WANDB_MODE WANDB_ENTITY; do
    if [[ -v "$name" ]]; then
        export "APPTAINERENV_${name}=${!name}"
    fi
done

# One container/torchrun launcher per node; torchrun starts one worker per GPU.
# Forward CUDA_VISIBLE_DEVICES inside each Slurm task, where Slurm sets it.
srun --export=ALL --ntasks="$NUM_NODES" --ntasks-per-node=1 -l bash -c '
    if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
        export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    fi
    exec apptainer "$@"
' bash exec --userns --cleanenv --nv \
    --bind "$REPO_ROOT:/workspace/TAGC" \
    --bind /share/bigyang:/share/bigyang \
    --pwd /workspace/TAGC/nanoGPT "$IMAGE" \
    /workspace/TAGC/.venv/bin/python -m torch.distributed.run \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NUM_NODES" \
    --rdzv_id "$SLURM_JOB_ID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    train.py config/train_gpt2.py > "${LOG_DIR}/exp.log" 2>&1
