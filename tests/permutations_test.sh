#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --job-name=bench_haps
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kj.benjamin90@gmail.com
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --output=logs/bench_haps.%j.log

set -euo pipefail

log_message() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"; }

# ----------------------------
# User-tunable args (comma-separated lists OK)
# ----------------------------
VARIANTS="${VARIANTS:-2000,8000,20000}"
PHENOTYPES="${PHENOTYPES:-50,200,500}"
SAMPLES="${SAMPLES:-256}"
ANCESTRIES="${SAMPLES:-2,3}"
COVARS="${COVARS:-6}"
DEVICE="${DEVICE:-auto}"     # auto|cpu|cuda
CSV_OUT="${CSV_OUT:-bench_haps_results.${SLURM_JOB_ID}.csv}"

# Where your repo lives; default to submission dir
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
SCRIPT_REL="./bench_haps.py"

log_message "**** Job starts ****"

log_message "**** Bridges-2 info ****"
echo "User: ${USER}"
echo "Job id: ${SLURM_JOBID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Node list: ${SLURM_NODELIST}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Cores/task: ${SLURM_CPUS_PER_TASK}"
echo "GPU(s): ${SLURM_GPUS}"

module purge
module load anaconda3/2024.10-1
module load cuda
module list

log_message "**** GPU info ****"
nvidia-smi || true

log_message "**** Loading conda environment ****"
# Adjust to your env path
conda activate /ocean/projects/bio250020p/shared/opt/env/AI_env

# Thread hygiene (avoid CPU oversubscription)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

log_message "**** Working directory ****"
cd "$WORKDIR"
echo "PWD: $(pwd)"

# Make results dir if you want to keep CSVs tidy
mkdir -p results

log_message "**** Run benchmark ****"
echo "Variants:   ${VARIANTS}"
echo "Phenotypes: ${PHENOTYPES}"
echo "Samples:    ${SAMPLES}"
echo "Ancestries: ${ANCESTRIES}"
echo "Covariates: ${COVARS}"
echo "Device:     ${DEVICE}"
echo "CSV out:    results/${CSV_OUT}"

# Use /usr/bin/time for a walltime summary (separate from the script's own timing)
{ /usr/bin/time -f "WALLCLOCK %E  MEM %M KB" \
python "${SCRIPT_REL}" \
  --variants "${VARIANTS}" \
  --phenotypes "${PHENOTYPES}" \
  --samples "${SAMPLES}" \
  --ancestries "${ANCESTRIES}" \
  --covars "${COVARS}" \
  --device "${DEVICE}" \
  --csv "results/${CSV_OUT}"; } 2>&1 | tee -a "results/bench_console.${SLURM_JOB_ID}.log"

log_message "**** Done. CSV saved to results/${CSV_OUT} ****"

conda deactivate
log_message "Job finished at: $(date)"

