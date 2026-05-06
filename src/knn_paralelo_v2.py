"""
KNN Dígitos — Versión PARALELA V2 (operaciones vectorizadas con NumPy)
═══════════════════════════════════════════════════════════════════════
Mejoras respecto a V1:
  ✔ knn_predict_vec: calcula todas las distancias del chunk en UNA sola
    operación de broadcasting NumPy, eliminando el doble bucle Python.
    O(n_test/p · n_train · d)  →  vectorizado sobre n_train y d.
  ✔ np.argpartition en lugar de argsort completo (O(n) vs O(n log n)).
  ✔ Estructura de comunicación MPI idéntica a V1 (bcast/scatter/gather).

Distancia vectorizada para un punto:
    dist = sqrt( sum((X_train - x)**2, axis=1) )   → shape (n_train,)

Distancia vectorizada para todo el chunk local:
    D[i,j] = sqrt( sum((X_train[j] - local_X[i])**2, axis=1) )
    D = sqrt( ((local_X[:,None,:] - X_train[None,:,:])**2).sum(axis=2) )
    shape: (local_n_test, n_train)

Uso:
    mpirun -n 4 python knn_paralelo_v2.py
"""
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import csv, os
 
CSV_PATH = "src/results_knn_v2.csv"
HEADERS  = [
    "version","it","n","p",
    "n_train","n_test","k",
    "t_total","t_compute","t_comm",
    "t_bcast","t_scatter","t_gather",
    "accuracy","flops","flops_per_sec"
]
 
def ensure_csv():
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH,"w",newline="") as f:
            csv.writer(f).writerow(HEADERS)
 
def append_row(row):
    ensure_csv()
    with open(CSV_PATH,"a",newline="") as f:
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
    n_train, n_test = len(X_train), len(X_test)
    chunks = np.array_split(X_test, size)
else:
    X_train=y_train=X_test=y_test=None
    n_train=n_test=None; chunks=None
 
# ── Paso 2: bcast ────────────────────────────────────────────────
_t0_comm = MPI.Wtime()
meta    = comm.bcast({"n_train":n_train,"n_test":n_test} if rank==0 else None, root=0)
n_train = meta["n_train"]; n_test = meta["n_test"]
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
t_bcast = MPI.Wtime() - _t0_comm
 
# ── Paso 3: scatter ──────────────────────────────────────────────
_t0_scatter  = MPI.Wtime()
local_X_test = comm.scatter(chunks, root=0)
t_scatter    = MPI.Wtime() - _t0_scatter
 
# ── Paso 4: cómputo matricial ─────────────────────────────────────
def knn_batch_vec(local_X, Xtr, ytr, k):
    """
    D[i,j] = ||local_X[i] - Xtr[j]||   via broadcasting
    Shape: (local_n, n_train)
    """
    diff     = local_X[:, np.newaxis, :] - Xtr[np.newaxis, :, :]  # (m,N,d)
    D        = np.sqrt((diff**2).sum(axis=2))                       # (m,N)
    k_idx    = np.argpartition(D, k, axis=1)[:, :k]                # (m,k)
    k_labels = ytr[k_idx]                                           # (m,k)
    return [Counter(row.tolist()).most_common(1)[0][0] for row in k_labels]
 
_t0_compute = MPI.Wtime()
local_preds = knn_batch_vec(local_X_test, X_train, y_train, k)
t_compute   = MPI.Wtime() - _t0_compute
 
# ── Paso 5: gather ───────────────────────────────────────────────
_t0_gather = MPI.Wtime()
all_preds  = comm.gather(local_preds, root=0)
t_gather   = MPI.Wtime() - _t0_gather
 
t_comm  = t_bcast + t_scatter + t_gather
t_total = t_compute + t_comm
 
# ── Evaluación y guardado ─────────────────────────────────────────
if rank == 0:
    y_pred  = [p for chunk in all_preds for p in chunk]
    acc     = float(np.mean(np.array(y_pred) == y_test))
    n_total = n_train + n_test
    flops   = int(n_test) * int(n_train) * (3*N_FEAT+1) ## dicho en el enunciado del proyecto
    fps     = flops / t_compute if t_compute > 0 else 0.0
 
    print(f"[V2] it={IT} n={n_total} p={size}  acc={acc:.4f}  "
          f"t_total={t_total:.4f}s  t_comp={t_compute:.4f}s  "
          f"t_comm={t_comm:.4f}s  GFLOP/s={fps/1e9:.4f}")
 
    append_row({
        "version":"v2","it":IT,"n":n_total,"p":size,
        "n_train":n_train,"n_test":n_test,"k":k,
        "t_total":round(t_total,6),"t_compute":round(t_compute,6),
        "t_comm":round(t_comm,6),
        "t_bcast":round(t_bcast,6),"t_scatter":round(t_scatter,6),
        "t_gather":round(t_gather,6),
        "accuracy":round(acc,4),
        "flops":flops,"flops_per_sec":round(fps,2),
    })
 
