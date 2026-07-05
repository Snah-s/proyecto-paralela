"""
KNN Dígitos — Versión PARALELA V3 (comunicación optimizada con buffers MPI)
════════════════════════════════════════════════════════════════════════════
Mejoras respecto a V2:
  ✔ comm.Bcast  (mayúscula) en lugar de comm.bcast:
      transmite buffers NumPy directamente por protocolo buffer, sin pickle.
  ✔ comm.Scatterv / comm.Gatherv:
      distribuyen/recolectan arrays NumPy contiguos en bruto, con soporte
      nativo para fragmentos de distinto tamaño (n_test % p ≠ 0).
  ✔ y_pred se recolecta como array NumPy entero (no lista de listas).
  ✔ Cómputo idéntico a V2 (matricial NumPy + argpartition).

Patrón de comunicación V3:
  ┌──────────────────────────────────────────────────────────┐
  │  Bcast  : int   k                  (scalar pickle, OK)   │
  │  Bcast  : ndarray  X_train  [N×d]  (buffer, sin pickle)  │
  │  Bcast  : ndarray  y_train  [N]    (buffer, sin pickle)  │
  │  Scatterv: ndarray X_test  [n×d]   (buffer contiguo)     │
  │  Gatherv : ndarray y_pred  [n]     (buffer contiguo)     │
  └──────────────────────────────────────────────────────────┘

FIXES aplicados vs versión con heap-corruption:
  ─────────────────────────────────────────────────────────────────
  BUG 1 — Scatterv receive buffer 2D:
    ANTES: local_X_test = np.empty((local_n, N_FEAT))  ← 2D, puede
           provocar escritura fuera de rango en algunos builds de MPI.
    FIX  : local_X_flat = np.empty(local_n * N_FEAT)   ← 1D explícito,
           luego reshape.

  BUG 2 — sendbuf=[None, counts, displ, dtype] en no-root:
    ANTES: [X_test_flat, counts, displ, MPI.DOUBLE] con X_test_flat=None
           → mpi4py intenta desreferenciar el puntero → heap corruption.
    FIX  : pasar None plano en no-root; mpi4py lo ignora correctamente.

  Mismo patrón aplicado a Gatherv (recvbuf=None en no-root).
  ─────────────────────────────────────────────────────────────────

Uso:
    mpirun -n 4 python knn_paralelo_v3.py
"""
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import csv, os

CSV_PATH = "src/results_knn_v3.csv"
HEADERS  = [
    "version","it","n","p",
    "n_train","n_test","k",
    "t_total","t_compute","t_comm",
    "t_bcast","t_scatter","t_gather",
    "accuracy","flops","flops_per_sec"
]

def ensure_csv():
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HEADERS)

def append_row(row):
    ensure_csv()
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([row[h] for h in HEADERS])

# ── MPI init ──────────────────────────────────────────────────────
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
size   = comm.Get_size()
N_FEAT = 64
k      = 3
IT        = int(os.environ.get("IT",        1))
DATA_SIZE = int(os.environ.get("DATA_SIZE", 1797))
DATA_DIR  = "src/data"

# ── Paso 1: Raíz carga datos ──────────────────────────────────────
if rank == 0:
    npz = os.path.join(DATA_DIR, f"digits_n{DATA_SIZE}.npz")
    if os.path.isfile(npz):
        d       = np.load(npz)
        X_train = d["X_train"].astype(np.float64)
        X_test  = d["X_test"].astype(np.float64)
        y_train = d["y_train"].astype(np.int32)
        y_test  = d["y_test"].astype(np.int32)
    else:
        digits  = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        X_train = X_train.astype(np.float64)
        X_test  = X_test.astype(np.float64)
        y_train = y_train.astype(np.int32)
        y_test  = y_test.astype(np.int32)

    n_train     = int(X_train.shape[0])
    n_test      = int(X_test.shape[0])
    # FIX: 1-D contiguo desde el inicio
    X_test_flat = np.ascontiguousarray(X_test.ravel(), dtype=np.float64)
else:
    X_train = y_train = X_test = y_test = None
    n_train = n_test = 0
    X_test_flat = None

# ── Paso 2: Bcast metadatos + datos de entrenamiento ─────────────
_t0_comm = MPI.Wtime()

meta    = comm.bcast({"n_train": n_train, "n_test": n_test} if rank == 0 else None, root=0)
n_train = meta["n_train"]
n_test  = meta["n_test"]

# Bcast por protocolo buffer (sin pickle) ─────────────────────────
if rank != 0:
    X_train = np.empty((n_train, N_FEAT), dtype=np.float64)
    y_train = np.empty(n_train,           dtype=np.int32)

comm.Bcast(X_train, root=0)
comm.Bcast(y_train, root=0)
t_bcast = MPI.Wtime() - _t0_comm

# ── Paso 3: Scatterv de X_test ────────────────────────────────────
# Cuántos puntos recibe cada proceso (distribución con resto)
counts     = np.array(
    [n_test // size + (1 if i < n_test % size else 0) for i in range(size)],
    dtype=np.int32
)
local_n    = int(counts[rank])
displ_rows = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int32)

# FIX 1: receive buffer 1-D explícito (sin 2D que causaba corrupción)
local_X_flat = np.empty(local_n * N_FEAT, dtype=np.float64)

# FIX 2: sendbuf=None en no-root (no [None, counts, displ, dtype])
if rank == 0:
    sendbuf_scatter = [
        X_test_flat,
        (counts * N_FEAT).tolist(),
        (displ_rows * N_FEAT).tolist(),
        MPI.DOUBLE
    ]
else:
    sendbuf_scatter = None   # ← ignorado por MPI en no-root

_t0_scatter = MPI.Wtime()
comm.Scatterv(
    sendbuf_scatter,
    [local_X_flat, int(local_n * N_FEAT), MPI.DOUBLE],
    root=0
)
t_scatter = MPI.Wtime() - _t0_scatter

# Reshape 2D después de recibir (ya sin riesgo)
local_X_test = local_X_flat.reshape((local_n, N_FEAT))

# ── Paso 4: Cómputo matricial ─────────────────────────────────────
def knn_batch_vec(local_X, Xtr, ytr, k):
    diff     = local_X[:, np.newaxis, :] - Xtr[np.newaxis, :, :]
    D        = np.sqrt((diff ** 2).sum(axis=2))
    k_idx    = np.argpartition(D, k, axis=1)[:, :k]
    k_labels = ytr[k_idx]
    return np.array(
        [Counter(row.tolist()).most_common(1)[0][0] for row in k_labels],
        dtype=np.int32
    )

_t0_compute = MPI.Wtime()
local_preds = knn_batch_vec(local_X_test, X_train, y_train, k)
t_compute   = MPI.Wtime() - _t0_compute

# ── Paso 5: Gatherv de predicciones ──────────────────────────────
# FIX: recvbuf=None en no-root (misma razón que sendbuf en Scatterv)
if rank == 0:
    y_pred_buf = np.empty(n_test, dtype=np.int32)
    recvbuf_gather = [
        y_pred_buf,
        counts.tolist(),
        displ_rows.tolist(),
        MPI.INT
    ]
else:
    y_pred_buf     = None
    recvbuf_gather = None   # ← ignorado por MPI en no-root

_t0_gather = MPI.Wtime()
comm.Gatherv(
    [local_preds, int(local_n), MPI.INT],
    recvbuf_gather,
    root=0
)
t_gather = MPI.Wtime() - _t0_gather

t_comm  = t_bcast + t_scatter + t_gather
t_total = t_compute + t_comm

# ── Evaluación y guardado (solo raíz) ────────────────────────────
if rank == 0:
    acc     = float(np.mean(y_pred_buf == y_test))
    n_total = n_train + n_test
    flops   = int(n_test) * int(n_train) * (3*N_FEAT+1) ## dicho en el enunciado del proyecto
    fps     = flops / t_compute if t_compute > 0 else 0.0

    print(f"[V3] it={IT} n={n_total} p={size}  acc={acc:.4f}  "
          f"t_total={t_total:.4f}s  t_comp={t_compute:.4f}s  "
          f"t_comm={t_comm:.4f}s  GFLOP/s={fps/1e9:.4f}")

    append_row({
        "version": "v3", "it": IT, "n": n_total, "p": size,
        "n_train": n_train, "n_test": n_test, "k": k,
        "t_total":       round(t_total,   6),
        "t_compute":     round(t_compute, 6),
        "t_comm":        round(t_comm,    6),
        "t_bcast":       round(t_bcast,   6),
        "t_scatter":     round(t_scatter, 6),
        "t_gather":      round(t_gather,  6),
        "accuracy":      round(acc,       4),
        "flops":         flops,
        "flops_per_sec": round(fps,       2),
    })