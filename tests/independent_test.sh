#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --job-name=bench_indep
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kj.benjamin90@gmail.com
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --output=logs/bench_independent.%j.log

set -euo pipefail

log_message() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"; }

# ----------------------------
# User-tunable args (comma-separated lists OK)
# ----------------------------
VARIANTS="${VARIANTS:-2000,8000,20000}"
PHENOTYPES="${PHENOTYPES:-50,200,500}"
SAMPLES="${SAMPLES:-256}"
ANCESTRIES="${ANCESTRIES:-0,2,3}"   # 0 disables haplotypes
NPERMS="${NPERMS:-200}"
COVARS="${COVARS:-6}"
FDRS="${FDRS:-0.05}"
MAFS="${MAFS:-0.00}"
SEEDS="${SEEDS:-13}"
RANDOM_TIEBREAK="${RANDOM_TIEBREAK:-0}"  # 1/true to enable
DEVICE="${DEVICE:-auto}"                  # auto|cpu|cuda
CSV_OUT="${CSV_OUT:-bench_independent_results.${SLURM_JOB_ID}.csv}"

# Where your repo lives; default to submission dir
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
SCRIPT_REL="./bench_independent.py"

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

# Thread hygiene
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

log_message "**** Working directory ****"
cd "$WORKDIR"
echo "PWD: $(pwd)"

mkdir -p results logs

log_message "**** Sweep settings ****"
echo "Variants:     ${VARIANTS}"
echo "Phenotypes:   ${PHENOTYPES}"
echo "Samples:      ${SAMPLES}"
echo "Ancestries:   ${ANCESTRIES}"
echo "Permutations: ${NPERMS}"
echo "Covariates:   ${COVARS}"
echo "FDRs:         ${FDRS}"
echo "MAFs:         ${MAFS}"
echo "Seeds:        ${SEEDS}"
echo "Random tie?   ${RANDOM_TIEBREAK}"
echo "Device:       ${DEVICE}"
echo "CSV out:      results/${CSV_OUT}"

# Parse comma-separated lists into arrays
IFS=',' read -r -a VAR_ARR <<< "${VARIANTS}"
IFS=',' read -r -a PHENO_ARR <<< "${PHENOTYPES}"
IFS=',' read -r -a SAMP_ARR <<< "${SAMPLES}"
IFS=',' read -r -a ANC_ARR <<< "${ANCESTRIES}"
IFS=',' read -r -a NPERM_ARR <<< "${NPERMS}"
IFS=',' read -r -a COV_ARR <<< "${COVARS}"
IFS=',' read -r -a FDR_ARR <<< "${FDRS}"
IFS=',' read -r -a MAF_ARR <<< "${MAFS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

# Helper: append CSV (keep header only once)
append_csv () {
  local tmp_csv="$1"
  local final_csv="$2"
  if [[ ! -s "${final_csv}" ]]; then
    mv "${tmp_csv}" "${final_csv}"
  else
    tail -n +2 "${tmp_csv}" >> "${final_csv}"
    rm -f "${tmp_csv}"
  fi
}

# Random tiebreak flag
RAND_FLAG=""
case "${RANDOM_TIEBREAK,,}" in
  1|true|yes|y) RAND_FLAG="--random_tiebreak" ;;
esac

RUN_LOG="results/bench_console.${SLURM_JOB_ID}.log"
: > "${RUN_LOG}"

# Sweep all combinations
for v in "${VAR_ARR[@]}"; do
  for p in "${PHENO_ARR[@]}"; do
    for s in "${SAMP_ARR[@]}"; do
      for k in "${ANC_ARR[@]}"; do
        for n in "${NPERM_ARR[@]}"; do
          for c in "${COV_ARR[@]}"; do
            for f in "${FDR_ARR[@]}"; do
              for maf in "${MAF_ARR[@]}"; do
                for seed in "${SEED_ARR[@]}"; do
                  tmp_csv="results/${CSV_OUT}.tmp.$$.$RANDOM"
                  log_message "Run: V=${v} P=${p} S=${s} K=${k} Nperm=${n} C=${c} FDR=${f} MAF=${maf} SEED=${seed}"

                  { /usr/bin/time -f "WALLCLOCK %E  MEM %M KB" \
                    python "${SCRIPT_REL}" \
                      --variants "${v}" \
                      --phenotypes "${p}" \
                      --samples "${s}" \
                      --ancestries "${k}" \
                      --covars "${c}" \
                      --nperm "${n}" \
                      --fdr "${f}" \
                      --maf "${maf}" \
                      --device "${DEVICE}" \
                      --seed "${seed}" \
                      ${RAND_FLAG} \
                      --csv "${tmp_csv}"; } 2>&1 | tee -a "${RUN_LOG}"

                  append_csv "${tmp_csv}" "results/${CSV_OUT}"
                done
              done
            done
          done
        done
      done
    done
  done
done

log_message "**** Done. CSV saved to results/${CSV_OUT} ****"

conda deactivate
log_message "Job finished at: $(date)"
