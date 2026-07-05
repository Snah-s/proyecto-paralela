#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_experiments.sh — Experimentos KNN paralelo (proyecto FINAL)
# ═══════════════════════════════════════════════════════════════════════════════
#  Barre las 3 etapas MPI (loop|vec|buf) + la versión OMP (numba) sobre
#  p ∈ {1,2,4,8} y n ∈ {1797,5000,10000,15000,20000}, ITERS repeticiones.
#
#  • MPI  → src/results_mpi.csv   (una fila por paradigm,stage,it,n,p)
#  • OMP  → src/results_omp.csv
#
#  Resume automático: se omiten combinaciones (paradigm,stage,it,n,p) ya en CSV.
#  --oversubscribe se activa cuando p > núcleos físicos (aquí 4c/8t).
#
#  Uso:
#     ./src/run_experiments.sh              # sweep completo
#     FORCE=1 ./src/run_experiments.sh      # re-corre todo
#     ONLY=omp ./src/run_experiments.sh     # solo OMP  (o ONLY=mpi)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."          # raíz del repo

PYTHON="${PYTHON:-.venv/bin/python}"
MPIRUN="${MPIRUN:-mpirun}"
ITERS="${ITERS:-5}"
P_LIST=(1 2 4 8)
N_LIST=(1797 5000 10000 15000 20000)
STAGES=(loop vec buf)
FORCE="${FORCE:-0}"
ONLY="${ONLY:-all}"             # all | mpi | omp
CSV_MPI="src/results_mpi.csv"
CSV_OMP="src/results_omp.csv"

# núcleos físicos → slots de OpenMPI (para decidir --oversubscribe)
N_PHYS=$(lscpu | awk -F: '/^Core\(s\) per socket:/{c=$2} /^Socket\(s\):/{s=$2} END{print (c+0)*(s+0)}')
[[ -z "$N_PHYS" || "$N_PHYS" -eq 0 ]] && N_PHYS=$(( $($PYTHON -c 'import os;print(os.cpu_count())') / 2 ))

echo "── Config ──────────────────────────────────────────"
echo "  python : $PYTHON"
echo "  mpirun : $($MPIRUN --version 2>&1 | head -1)"
echo "  físicos: $N_PHYS   ITERS=$ITERS   ONLY=$ONLY   FORCE=$FORCE"
echo "  stages : ${STAGES[*]}   p: ${P_LIST[*]}   n: ${N_LIST[*]}"
echo "────────────────────────────────────────────────────"

# ¿existe ya la fila (col1=paradigm, col2=stage, col3=it, col4=n, col5=p)?
row_exists() {  # $1=csv $2=paradigm $3=stage $4=it $5=n $6=p
    [[ "$FORCE" == "1" ]] && return 1
    [[ ! -f "$1" ]] && return 1
    awk -F',' -v pa="$2" -v st="$3" -v it="$4" -v n="$5" -v p="$6" \
        'NR>1 && $1==pa && $2==st && $3==it && $4==n && $5==p {found=1; exit} END{exit !found}' "$1"
}

t0=$(date +%s)

# ── MPI ──────────────────────────────────────────────────────────────────────
if [[ "$ONLY" == "all" || "$ONLY" == "mpi" ]]; then
  for stage in "${STAGES[@]}"; do
    for n in "${N_LIST[@]}"; do
      for p in "${P_LIST[@]}"; do
        over=""; (( p > N_PHYS )) && over="--oversubscribe"
        for it in $(seq 1 "$ITERS"); do
          if row_exists "$CSV_MPI" "mpi" "$stage" "$it" "$n" "$p"; then
            echo "  skip  mpi/$stage n=$n p=$p it=$it"
          else
            STAGE=$stage DATA_SIZE=$n IT=$it PYTHONWARNINGS="ignore" \
              $MPIRUN $over -n "$p" "$PYTHON" src/knn_paralelo.py \
              || { echo "  ✗ FALLO mpi/$stage n=$n p=$p it=$it"; }
          fi
        done
      done
    done
  done
fi

# ── OMP (numba) ──────────────────────────────────────────────────────────────
if [[ "$ONLY" == "all" || "$ONLY" == "omp" ]]; then
  for n in "${N_LIST[@]}"; do
    for p in "${P_LIST[@]}"; do
      for it in $(seq 1 "$ITERS"); do
        if row_exists "$CSV_OMP" "omp" "numba" "$it" "$n" "$p"; then
          echo "  skip  omp n=$n threads=$p it=$it"
        else
          THREADS=$p DATA_SIZE=$n IT=$it \
            "$PYTHON" src/knn_omp.py \
            || { echo "  ✗ FALLO omp n=$n threads=$p it=$it"; }
        fi
      done
    done
  done
fi

t1=$(date +%s)
echo "────────────────────────────────────────────────────"
echo "  Terminado en $(( (t1-t0)/60 ))m $(( (t1-t0)%60 ))s"
[[ -f "$CSV_MPI" ]] && echo "  $CSV_MPI : $(( $(wc -l < "$CSV_MPI") - 1 )) filas"
[[ -f "$CSV_OMP" ]] && echo "  $CSV_OMP : $(( $(wc -l < "$CSV_OMP") - 1 )) filas"
