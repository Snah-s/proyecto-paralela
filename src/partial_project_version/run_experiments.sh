#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  run_experiments.sh — Experimentos KNN paralelo
# ═══════════════════════════════════════════════════════════════════
#
#  ROOT CAUSE del problema "4 singletons":
#  ┌─────────────────────────────────────────────────────────────────┐
#  │  El `mpirun` en PATH es de MPICH (muestra "HYDRA build"),       │
#  │  pero mpi4py fue compilado con Open MPI (conda).                │
#  │  Son implementaciones distintas — MPICH lanza 4 singletons      │
#  │  independientes porque no puede inicializar el contexto MPI      │
#  │  que Open MPI espera. Cada proceso ve size=1.                   │
#  │                                                                  │
#  │  FIX: usar $CONDA_PREFIX/bin/mpirun (Open MPI de conda)         │
#  │  en lugar del mpirun del sistema (MPICH/SLURM).                 │
#  │                                                                  │
#  │  Verificar manualmente:                                          │
#  │    which mpirun              → puede ser MPICH del sistema       │
#  │    $CONDA_PREFIX/bin/mpirun  → Open MPI de conda ← usar este    │
#  └─────────────────────────────────────────────────────────────────┘
#
#  Uso:
#    chmod +x src/run_experiments.sh
#    ./src/run_experiments.sh                  # solo DATA_SIZE=1797
#    ALL_SIZES=1 ./src/run_experiments.sh      # todos los tamaños
#    FORCE=1     ./src/run_experiments.sh      # re-corre todo
#    VERSION=v3  ./src/run_experiments.sh      # solo una versión
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ──────────────────────────────────────────────
#  Seleccionar el mpirun correcto (Open MPI de conda)
#
#  Prioridad:
#    1. $CONDA_PREFIX/bin/mpirun  (Open MPI del entorno activo)
#    2. mpirun del PATH           (fallback, puede ser MPICH)
#
#  Si usas el mpirun del sistema (MPICH) con mpi4py compilado
#  para Open MPI obtienes 4 singletons en lugar de 1 job de 4.
# ──────────────────────────────────────────────
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/mpirun" ]]; then
    MPIRUN="${CONDA_PREFIX}/bin/mpirun"
else
    MPIRUN="mpirun"
fi

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
ITERS=5
P_LIST=(1 2 4 8)

if [[ -n "${VERSION:-}" ]]; then
    VERSIONS=("$VERSION")
else
    VERSIONS=(v1 v2 v3)
fi

if [[ "${ALL_SIZES:-0}" == "1" ]]; then
    DATA_SIZES=(1797 5000 10000 15000 20000)
else
    DATA_SIZES=(1797)
fi

SRC_DIR="src"
FORCE="${FORCE:-0}"
DATA_SCRIPT="${SRC_DIR}/knn_data_generate.py"

# ──────────────────────────────────────────────
#  Colores
# ──────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
CYN='\033[0;36m'; DIM='\033[2m'; NC='\033[0m'

log_section() {
    echo -e "\n${CYN}══════════════════════════════════════${NC}"
    echo -e "${CYN}  $1${NC}"
    echo -e "${CYN}══════════════════════════════════════${NC}"
}
log_ok()   { echo -e "  ${GRN}✓${NC} $1"; }
log_run()  { echo -e "  ${YLW}▶${NC} $1"; }
log_skip() { echo -e "  ${DIM}⏭ SKIP${NC} $1"; }
log_err()  { echo -e "  ${RED}✗${NC} $1"; }

# ──────────────────────────────────────────────
#  CSV por versión
# ──────────────────────────────────────────────
csv_for() { echo "${SRC_DIR}/results_knn_${1}.csv"; }

# ──────────────────────────────────────────────
#  Detectar núcleos físicos
# ──────────────────────────────────────────────
N_LOGICAL=$(python -c "import os; print(os.cpu_count())")
if command -v lscpu &>/dev/null; then
    N_PHYSICAL=$(lscpu | awk '
        /^Core\(s\) per socket:/  { cores=$NF }
        /^Socket\(s\):/           { sockets=$NF }
        END { print (cores+0) * (sockets+0) }
    ')
    [[ -z "$N_PHYSICAL" || "$N_PHYSICAL" -eq 0 ]] && N_PHYSICAL=$(( N_LOGICAL / 2 ))
else
    N_PHYSICAL=$(( N_LOGICAL / 2 ))
fi
N_PHYSICAL=$(( N_PHYSICAL > 0 ? N_PHYSICAL : 1 ))

# ──────────────────────────────────────────────
#  Resume helpers
# ──────────────────────────────────────────────
row_exists() {
    local ver=$1 it=$2 n=$3 p=$4
    local csv_file; csv_file=$(csv_for "$ver")
    [[ "$FORCE" == "1" ]]  && return 1
    [[ ! -f "$csv_file" ]] && return 1
    local found
    found=$(awk -F',' -v v="$ver" -v i="$it" -v n="$n" -v p="$p" \
        'NR>1 && $1==v && $2==i && $3==n && $4==p {print 1; exit}' \
        "$csv_file")
    [[ "$found" == "1" ]]
}

block_done_count() {
    local ver=$1 ds=$2 p=$3
    local csv_file; csv_file=$(csv_for "$ver")
    [[ ! -f "$csv_file" ]] && echo 0 && return
    awk -F',' -v v="$ver" -v n="$ds" -v p="$p" \
        'NR>1 && $1==v && $3==n && $4==p' "$csv_file" | wc -l
}

# ──────────────────────────────────────────────
#  Pre-checks
# ──────────────────────────────────────────────
log_section "Verificación del entorno"

python --version && log_ok "Python OK" \
    || { log_err "Python no encontrado"; exit 1; }

# Mostrar cuál mpirun se va a usar y verificar que es Open MPI
echo -e "  mpirun seleccionado: ${YLW}${MPIRUN}${NC}"
MPIRUN_VERSION=$("$MPIRUN" --version 2>&1 | head -1)
echo -e "  Versión: ${MPIRUN_VERSION}"

if echo "$MPIRUN_VERSION" | grep -qi "HYDRA"; then
    echo -e "  ${RED}⚠ ADVERTENCIA${NC}: Este mpirun es de MPICH (HYDRA)."
    echo -e "  mpi4py fue compilado con Open MPI — habrá conflicto."
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        echo -e "  Intentando con: ${CONDA_PREFIX}/bin/mpirun"
        MPIRUN="${CONDA_PREFIX}/bin/mpirun"
        MPIRUN_VERSION=$("$MPIRUN" --version 2>&1 | head -1)
        echo -e "  Nueva versión: ${MPIRUN_VERSION}"
    fi
fi

if echo "$MPIRUN_VERSION" | grep -qi "Open MPI\|OpenMPI"; then
    log_ok "mpirun es Open MPI ✓"
else
    log_err "mpirun no es Open MPI — posible conflicto con mpi4py"
    echo -e "  Solución manual:"
    echo -e "    conda install -c conda-forge openmpi mpi4py --force-reinstall"
    echo -e "  O usar explícitamente: MPIRUN=\$CONDA_PREFIX/bin/mpirun ./src/run_experiments.sh"
fi

python -c "from mpi4py import MPI; print('mpi4py', MPI.Get_library_version()[:40])" \
    && log_ok "mpi4py OK" \
    || { log_err "mpi4py no instalado"; exit 1; }

[[ -f "$DATA_SCRIPT" ]] \
    && log_ok "Script de datos: ${DATA_SCRIPT}" \
    || { log_err "No se encontró: ${DATA_SCRIPT}"; exit 1; }

echo -e ""
echo -e "  Hilos lógicos   : ${YLW}${N_LOGICAL}${NC}"
echo -e "  Núcleos físicos : ${YLW}${N_PHYSICAL}${NC}  (slots OpenMPI)"
echo -e "  --oversubscribe : se activa para p > ${N_PHYSICAL}"
echo -e "  Versiones       : ${YLW}${VERSIONS[*]}${NC}"
echo -e "  Tamaños n       : ${YLW}${DATA_SIZES[*]}${NC}"
echo -e "  Iteraciones     : ${YLW}${ITERS}${NC}  por (versión, p, n)"
[[ "$FORCE" == "1" ]] \
    && echo -e "  ${RED}FORCE=1${NC}: re-corriendo todo" \
    || echo -e "  ${GRN}RESUME activo${NC}: runs ya en CSV se omiten"

# Estado actual de CSVs
echo ""
for ver in "${VERSIONS[@]}"; do
    csv_file=$(csv_for "$ver")
    if [[ -f "$csv_file" && "$FORCE" != "1" ]]; then
        ROWS=$(( $(wc -l < "$csv_file") - 1 ))
        echo -e "  ${GRN}${csv_file}${NC}: ${YLW}${ROWS}${NC} filas"
        awk -F',' 'NR>1 {print $4, $3}' "$csv_file" \
            | sort | uniq -c \
            | awk '{printf "      %3d runs  p=%-2s  n=%s\n", $1,$2,$3}'
    else
        echo -e "  ${DIM}${csv_file}${NC}: no existe aún"
    fi
done

# ──────────────────────────────────────────────
#  Generar datasets si no existen
# ──────────────────────────────────────────────
log_section "Datasets"
mkdir -p "${SRC_DIR}/data"

if [[ ! -f "${SRC_DIR}/data/digits_n1797.npz" ]]; then
    log_run "Generando datasets con ${DATA_SCRIPT} ..."
    python "$DATA_SCRIPT" \
        || { log_err "Falló la generación de datasets"; exit 1; }
    log_ok "Datasets generados"
else
    log_ok "Datasets ya existen en ./${SRC_DIR}/data/"
fi

for ds in "${DATA_SIZES[@]}"; do
    npz="${SRC_DIR}/data/digits_n${ds}.npz"
    [[ -f "$npz" ]] || { log_err "Falta: ${npz}. Corre: python ${DATA_SCRIPT}"; exit 1; }
done
log_ok "Todos los datasets verificados"

# ──────────────────────────────────────────────
#  Función de ejecución paralela
# ──────────────────────────────────────────────
run_parallel() {
    local ver=$1 ds=$2 p=$3
    local script="${SRC_DIR}/knn_paralelo_${ver}.py"

    [[ -f "$script" ]] || { log_err "No se encontró: ${script}"; exit 1; }

    local oversubscribe=""
    if (( p > N_PHYSICAL )); then
        oversubscribe="--oversubscribe"
        echo -e "  ${YLW}⚠ --oversubscribe${NC}: p=${p} > físicos=${N_PHYSICAL}"
    fi

    local ran=0 skipped=0

    for it in $(seq 1 "$ITERS"); do
        if row_exists "$ver" "$it" "$ds" "$p"; then
            log_skip "[${ver^^}] p=${p}  it=${it}/${ITERS}  n=${ds}"
            (( skipped++ )) || true
        else
            log_run "[${ver^^}] p=${p}  it=${it}/${ITERS}  n=${ds}"
            IT=$it DATA_SIZE=$ds PYTHONWARNINGS="ignore::RuntimeWarning" \
                "$MPIRUN" $oversubscribe -n "$p" python "$script" \
                || { log_err "Falló: ${script} p=${p} it=${it}"; exit 1; }
            (( ran++ )) || true
        fi
    done

    if (( skipped > 0 && ran == 0 )); then
        log_ok "[${ver^^}] p=${p} n=${ds} — todos ya completados (${skipped}/${ITERS})"
    elif (( skipped > 0 )); then
        log_ok "[${ver^^}] p=${p} n=${ds} — ${ran} nuevos + ${skipped} ya existían"
    fi
}

# ──────────────────────────────────────────────
#  Loop principal
# ──────────────────────────────────────────────
TOTAL_NEW=0
TOTAL_SKIP=0
START_ALL=$(date +%s)

for ver in "${VERSIONS[@]}"; do
    log_section "VERSIÓN ${ver^^}"

    for ds in "${DATA_SIZES[@]}"; do
        log_section "${ver^^} — n=${ds}"

        for p in "${P_LIST[@]}"; do
            DONE=$(block_done_count "$ver" "$ds" "$p")

            if (( DONE >= ITERS )); then
                log_skip "${ver^^} p=${p} n=${ds} — ${ITERS}/${ITERS} ya en CSV"
                TOTAL_SKIP=$(( TOTAL_SKIP + ITERS ))
            else
                run_parallel "$ver" "$ds" "$p"
                TOTAL_NEW=$(( TOTAL_NEW   + ITERS - DONE ))
                TOTAL_SKIP=$(( TOTAL_SKIP + DONE ))
            fi
        done

        log_ok "${ver^^} n=${ds} ✓"
    done

    log_ok "VERSIÓN ${ver^^} completada"
done

END_ALL=$(date +%s)
ELAPSED=$(( END_ALL - START_ALL ))

# ──────────────────────────────────────────────
#  Resumen final
# ──────────────────────────────────────────────
log_section "Resumen final"
echo -e "  mpirun usado    : ${YLW}${MPIRUN}${NC}"
echo -e "  Tiempo total    : ${YLW}${ELAPSED} s  ($(( ELAPSED/60 ))m $(( ELAPSED%60 ))s)${NC}"
echo -e "  Runs nuevos     : ${GRN}${TOTAL_NEW}${NC}"
echo -e "  Runs omitidos   : ${DIM}${TOTAL_SKIP}${NC}"
echo ""

EXPECTED=$(( ITERS * ${#P_LIST[@]} * ${#DATA_SIZES[@]} ))
for ver in "${VERSIONS[@]}"; do
    csv_file=$(csv_for "$ver")
    if [[ -f "$csv_file" ]]; then
        ROWS=$(( $(wc -l < "$csv_file") - 1 ))
        STATUS="${GRN}"
        (( ROWS < EXPECTED )) && STATUS="${YLW}"
        echo -e "  ${STATUS}${csv_file}${NC}: ${YLW}${ROWS}${NC} / ${EXPECTED} filas esperadas"
    fi
done

echo ""
log_ok "¡Listo! → jupyter notebook experiments_analysis.ipynb"