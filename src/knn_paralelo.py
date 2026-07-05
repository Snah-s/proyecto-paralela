"""
knn_paralelo.py — KNN paralelo con MPI (mpi4py), UN SOLO código en 3 etapas.
═══════════════════════════════════════════════════════════════════════════════
El desarrollo del proyecto parcial usaba 3 archivos (knn_paralelo_v1/v2/v3.py).
Aquí se consolidan en UN único programa cuyo comportamiento se selecciona con la
variable de entorno STAGE. Las tres etapas registran el desarrollo incremental
(inciso b del enunciado: "registre el desarrollo en ≥3 pasos"):

  STAGE=loop  →  Etapa 1 — baseline: distancia con bucle Python + colectivas
                 pickle (comm.bcast / comm.scatter / comm.gather).
  STAGE=vec   →  Etapa 2 — cómputo vectorizado NumPy (broadcasting por lotes)
                 + argpartition; misma comunicación pickle.
  STAGE=buf   →  Etapa 3 — comunicación por buffers (comm.Bcast / Scatterv /
                 Gatherv) sin serialización pickle; cómputo idéntico a la etapa 2.

Todas las etapas son MATEMÁTICAMENTE equivalentes al secuencial: la accuracy
es idéntica e invariante en p (validación del inciso b).

── Modelo PRAM (CREW) ──────────────────────────────────────────────────────────
  Concurrent Read : todos los procesos reciben X_train por bcast (solo lectura).
  Exclusive Write : cada proceso predice su partición disjunta de X_test.
  Comunicación    : colectivas de árbol binomial  ⇒  T_comm = Θ(log p · (α+βm)).

── Memoria acotada ─────────────────────────────────────────────────────────────
  El broadcasting 3-D (n_local × n_train × d) explotaría la RAM para n grande
  (~32 GB con n=20000). Se calcula por LOTES de filas de test para acotar el pico
  de memoria a ~256 MB sin alterar el resultado numérico.

Uso:
    STAGE=vec DATA_SIZE=5000 IT=1 mpirun -n 4 python src/knn_paralelo.py
"""
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import csv, os

# ── Configuración ────────────────────────────────────────────────────────────
CSV_PATH   = "src/results_mpi.csv"
HEADERS    = [
    "paradigm", "stage", "it", "n", "p",
    "n_train", "n_test", "k",
    "t_total", "t_compute", "t_comm",
    "t_bcast", "t_scatter", "t_gather",
    "accuracy", "flops", "flops_per_sec",
]
N_FEAT     = 64
K          = 3
STAGE      = os.environ.get("STAGE", "loop").lower()   # loop | vec | buf
IT         = int(os.environ.get("IT", 1))
DATA_SIZE  = int(os.environ.get("DATA_SIZE", 1797))
DATA_DIR   = "src/data"
# FLOPs por distancia euclidiana en R^d: d restas + d mult + (d-1) sumas + 1 raíz = 3d
FLOP_PER_DIST = 3 * N_FEAT
# Pico de memoria objetivo para el tensor por lotes (bytes)
BATCH_TARGET_BYTES = 256 * 1024 * 1024

assert STAGE in ("loop", "vec", "buf"), f"STAGE inválido: {STAGE}"


def ensure_csv():
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HEADERS)


def append_row(row):
    ensure_csv()
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([row[h] for h in HEADERS])


def majority_vote(labels_row):
    """Voto mayoritario sobre k etiquetas (0..9) vía bincount (rápido y estable)."""
    return int(np.bincount(labels_row, minlength=10).argmax())


# ── Etapa 1: distancia con bucle Python (baseline) ───────────────────────────
def knn_loop(local_X, Xtr, ytr, k):
    preds = np.empty(local_X.shape[0], dtype=np.int32)
    for i in range(local_X.shape[0]):
        dists = np.sqrt(((Xtr - local_X[i]) ** 2).sum(axis=1))
        k_idx = np.argpartition(dists, k)[:k]
        preds[i] = majority_vote(ytr[k_idx])
    return preds


# ── Etapas 2 y 3: distancia vectorizada por lotes (memoria acotada) ──────────
def knn_vec_batched(local_X, Xtr, ytr, k):
    m = local_X.shape[0]
    n_train = Xtr.shape[0]
    preds = np.empty(m, dtype=np.int32)
    if m == 0:
        return preds
    # tamaño de lote para acotar el pico de memoria del tensor (B × n_train × d)
    per_row = n_train * N_FEAT * 8
    batch = int(np.clip(BATCH_TARGET_BYTES // max(per_row, 1), 8, 1024))
    for s in range(0, m, batch):
        e = min(s + batch, m)
        # broadcasting: (b, n_train, d) → distancia euclidiana (b, n_train)
        diff = local_X[s:e, None, :] - Xtr[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=2))
        k_idx = np.argpartition(D, k, axis=1)[:, :k]        # O(n_train) por fila
        k_lab = ytr[k_idx]                                   # (b, k)
        for j in range(e - s):
            preds[s + j] = majority_vote(k_lab[j])
    return preds


# ══════════════════════════════════════════════════════════════════════════════
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── Paso 1: la raíz carga los datos ──────────────────────────────────────────
if rank == 0:
    npz_path = os.path.join(DATA_DIR, f"digits_n{DATA_SIZE}.npz")
    if os.path.isfile(npz_path):
        d = np.load(npz_path)
        X_train = d["X_train"].astype(np.float64)
        X_test  = d["X_test"].astype(np.float64)
        y_train = d["y_train"].astype(np.int32)
        y_test  = d["y_test"].astype(np.int32)
    else:
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
        )
        X_train = X_train.astype(np.float64); X_test = X_test.astype(np.float64)
        y_train = y_train.astype(np.int32);   y_test = y_test.astype(np.int32)
    n_train, n_test = int(X_train.shape[0]), int(X_test.shape[0])
else:
    X_train = y_train = X_test = y_test = None
    n_train = n_test = 0

# ── Paso 2: bcast de metadatos + datos de entrenamiento (Concurrent Read) ────
_t0 = MPI.Wtime()
meta = comm.bcast({"n_train": n_train, "n_test": n_test} if rank == 0 else None, root=0)
n_train, n_test = meta["n_train"], meta["n_test"]

if STAGE == "buf":
    # comunicación por buffers: sin pickle
    if rank != 0:
        X_train = np.empty((n_train, N_FEAT), dtype=np.float64)
        y_train = np.empty(n_train,           dtype=np.int32)
    comm.Bcast(X_train, root=0)
    comm.Bcast(y_train, root=0)
else:
    # comunicación pickle (etapas loop y vec)
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
t_bcast = MPI.Wtime() - _t0

# ── Paso 3: scatter de X_test ────────────────────────────────────────────────
# reparto con resto: counts[i] = floor(n_test/p) + (i < n_test % p)
counts = np.array([n_test // size + (1 if i < n_test % size else 0)
                   for i in range(size)], dtype=np.int32)
local_n = int(counts[rank])
displ   = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int32)

_t0 = MPI.Wtime()
if STAGE == "buf":
    if rank == 0:
        X_test_flat = np.ascontiguousarray(X_test.ravel(), dtype=np.float64)
        sendbuf = [X_test_flat, (counts * N_FEAT), (displ * N_FEAT), MPI.DOUBLE]
    else:
        sendbuf = None
    local_flat = np.empty(local_n * N_FEAT, dtype=np.float64)
    comm.Scatterv(sendbuf, [local_flat, local_n * N_FEAT, MPI.DOUBLE], root=0)
    local_X = local_flat.reshape((local_n, N_FEAT))
else:
    chunks  = np.array_split(X_test, size) if rank == 0 else None
    local_X = comm.scatter(chunks, root=0)
t_scatter = MPI.Wtime() - _t0

# ── Paso 4: cómputo local (Exclusive Write) ──────────────────────────────────
_t0 = MPI.Wtime()
if STAGE == "loop":
    local_preds = knn_loop(local_X, X_train, y_train, K)
else:
    local_preds = knn_vec_batched(local_X, X_train, y_train, K)
t_compute = MPI.Wtime() - _t0

# ── Paso 5: gather de predicciones ───────────────────────────────────────────
_t0 = MPI.Wtime()
if STAGE == "buf":
    if rank == 0:
        y_pred = np.empty(n_test, dtype=np.int32)
        recvbuf = [y_pred, counts, displ, MPI.INT]
    else:
        y_pred, recvbuf = None, None
    comm.Gatherv([local_preds, local_n, MPI.INT], recvbuf, root=0)
else:
    gathered = comm.gather(local_preds, root=0)
    y_pred = np.concatenate(gathered) if rank == 0 else None
t_gather = MPI.Wtime() - _t0

t_comm  = t_bcast + t_scatter + t_gather
t_total = t_compute + t_comm

# ── Evaluación y guardado (solo raíz) ────────────────────────────────────────
if rank == 0:
    acc   = float(np.mean(y_pred == y_test))
    flops = int(n_test) * int(n_train) * FLOP_PER_DIST
    fps   = flops / t_compute if t_compute > 0 else 0.0
    print(f"[MPI/{STAGE}] it={IT} n={n_train + n_test} p={size} acc={acc:.4f} "
          f"t_total={t_total:.4f}s t_comp={t_compute:.4f}s t_comm={t_comm:.4f}s "
          f"GFLOP/s={fps/1e9:.4f}")
    append_row({
        "paradigm": "mpi", "stage": STAGE, "it": IT, "n": n_train + n_test, "p": size,
        "n_train": n_train, "n_test": n_test, "k": K,
        "t_total": round(t_total, 6), "t_compute": round(t_compute, 6),
        "t_comm": round(t_comm, 6), "t_bcast": round(t_bcast, 6),
        "t_scatter": round(t_scatter, 6), "t_gather": round(t_gather, 6),
        "accuracy": round(acc, 4), "flops": flops, "flops_per_sec": round(fps, 2),
    })
