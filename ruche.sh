#!/bin/bash
#SBATCH --job-name=jepacode
#SBATCH --partition=cpu_med
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load llvm 2>/dev/null || true
for t in opt clang llc llvm-size; do command -v "$t" >/dev/null || { echo "$t not on PATH" >&2; exit 1; }; done
source "${VENV:-/gpfs/workdir/$USER/venvs/jepacode}/bin/activate"
python run.py "$@"
