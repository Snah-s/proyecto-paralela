"""
knn_omp.py — KNN paralelo de MEMORIA COMPARTIDA (analógo OMP) con numba prange.
═══════════════════════════════════════════════════════════════════════════════
Contraparte de knn_paralelo.py (MPI) para la comparación de paradigmas del
inciso a) del enunciado: "¿distribuido o compartido? ¿OMP o MPI?".

En Python el GIL impide el paralelismo de hilos real; el análogo honesto de
`#pragma omp parallel for` es numba con `@njit(parallel=True)` + `prange`, que
emite HILOS OpenMP reales sobre código nativo, SIN GIL y COMPARTIENDO X_train
(memoria compartida: una sola copia, sin bcast, sin serialización pickle).

Diferencia clave con MPI:
  • MPI      : replica X_train en cada proceso (bcast) + serialización + Θ(log p).
  • OMP/numba: X_train vive una sola vez en memoria; los hilos la leen de forma
               concurrente (PRAM CREW real). No hay T_comm.

El bucle externo `prange` reparte los puntos de test entre hilos (escritura
exclusiva en `preds[i]`), idéntico patrón de paralelismo de datos que MPI pero
sin coste de comunicación.

Uso:
    THREADS=4 DATA_SIZE=5000 IT=1 python src/knn_omp.py
"""
import numpy as np
from numba import njit, prange, set_num_threads
import csv, os, time

CSV_PATH  = "src/results_omp.csv"
HEADERS   = [
    "paradigm", "stage", "it", "n", "p",
    "n_train", "n_test", "k",
    "t_total", "t_compute", "t_comm",
    "t_bcast", "t_scatter", "t_gather",
    "accuracy", "flops", "flops_per_sec",
]
N_FEAT    = 64
K         = 3
THREADS   = int(os.environ.get("THREADS", 1))
IT        = int(os.environ.get("IT", 1))
DATA_SIZE = int(os.environ.get("DATA_SIZE", 1797))
DATA_DIR  = "src/data"
FLOP_PER_DIST = 3 * N_FEAT   # d restas + d mult + (d-1) sumas + 1 raíz ≈ 3d


def ensure_csv():
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HEADERS)


def append_row(row):
    ensure_csv()
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([row[h] for h in HEADERS])


@njit(parallel=True, fastmath=True, cache=True)
def knn_numba(X_test, X_train, y_train, k):
    """KNN con selección de k-menores en línea. prange = paralelismo por hilos.
    X_train es COMPARTIDA por todos los hilos (lectura concurrente, sin réplica)."""
    n_test  = X_test.shape[0]
    n_train = X_train.shape[0]
    d       = X_test.shape[1]
    preds   = np.empty(n_test, dtype=np.int32)
    for i in prange(n_test):                       # ← reparto de test entre hilos OMP
        best_d = np.full(k, 1e18)                   # k distancias² menores
        best_l = np.full(k, -1, dtype=np.int32)     # etiquetas asociadas
        for j in range(n_train):
            dist = 0.0
            for f in range(d):
                diff = X_test[i, f] - X_train[j, f]
                dist += diff * diff                 # distancia² (evita sqrt)
            # mantener los k menores: reemplazar el mayor actual si procede
            maxpos = 0
            for t in range(1, k):
                if best_d[t] > best_d[maxpos]:
                    maxpos = t
            if dist < best_d[maxpos]:
                best_d[maxpos] = dist
                best_l[maxpos] = y_train[j]
        # voto mayoritario (10 clases)
        cnt = np.zeros(10, dtype=np.int32)
        for t in range(k):
            cnt[best_l[t]] += 1
        preds[i] = np.argmax(cnt)
    return preds


# ── Carga de datos ───────────────────────────────────────────────────────────
npz_path = os.path.join(DATA_DIR, f"digits_n{DATA_SIZE}.npz")
if os.path.isfile(npz_path):
    d = np.load(npz_path)
    X_train = d["X_train"].astype(np.float64); X_test = d["X_test"].astype(np.float64)
    y_train = d["y_train"].astype(np.int32);   y_test = d["y_test"].astype(np.int32)
else:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target)
    X_train = X_train.astype(np.float64); X_test = X_test.astype(np.float64)
    y_train = y_train.astype(np.int32);   y_test = y_test.astype(np.int32)

n_train, n_test = int(X_train.shape[0]), int(X_test.shape[0])
set_num_threads(THREADS)

# ── Warmup: fuerza la compilación JIT fuera de la medición ───────────────────
_ = knn_numba(X_test[:8], X_train[:16], y_train[:16], K)

# ── Medición del cómputo (memoria compartida, sin comunicación) ──────────────
_t0 = time.perf_counter()
y_pred = knn_numba(X_test, X_train, y_train, K)
t_compute = time.perf_counter() - _t0

t_comm  = 0.0                    # memoria compartida: sin bcast/scatter/gather
t_total = t_compute + t_comm

acc   = float(np.mean(y_pred == y_test))
flops = int(n_test) * int(n_train) * FLOP_PER_DIST
fps   = flops / t_compute if t_compute > 0 else 0.0
print(f"[OMP/numba] it={IT} n={n_train + n_test} threads={THREADS} acc={acc:.4f} "
      f"t_total={t_total:.4f}s t_comp={t_compute:.4f}s GFLOP/s={fps/1e9:.4f}")
append_row({
    "paradigm": "omp", "stage": "numba", "it": IT, "n": n_train + n_test, "p": THREADS,
    "n_train": n_train, "n_test": n_test, "k": K,
    "t_total": round(t_total, 6), "t_compute": round(t_compute, 6),
    "t_comm": 0.0, "t_bcast": 0.0, "t_scatter": 0.0, "t_gather": 0.0,
    "accuracy": round(acc, 4), "flops": flops, "flops_per_sec": round(fps, 2),
})
