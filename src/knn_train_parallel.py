"""
knn_train_parallel.py — KNN paralelo PARTICIONANDO EL TRAIN (estrategia rival).
═══════════════════════════════════════════════════════════════════════════════
Este programa existe para PROBAR EMPÍRICAMENTE, no solo afirmar, que en nuestro
contexto (un nodo, X_train cabe en memoria) es mejor paralelizar TEST que TRAIN.

Contraste de estrategias
────────────────────────
  knn_paralelo.py  (TEST) : bcast(X_train) a todos + scatter(X_test).
                            Cada proceso ve TODO el train ⇒ produce la etiqueta
                            FINAL de su bloque de test. gather = concatenar ŷ.
                            Embarazosamente paralelo, sin fusión de resultados.

  knn_train_parallel.py (TRAIN) : bcast(X_test) a todos + scatter(X_train).
                            Cada proceso ve SOLO un fragmento del train ⇒ produce
                            k-vecinos PARCIALES por punto de test. Hace falta una
                            REDUCCIÓN global: recolectar los p·k candidatos por
                            punto y fusionarlos (t_merge) para obtener los k reales.

Coste de comunicación (por qué TRAIN pierde en un nodo)
───────────────────────────────────────────────────────
  • TEST : gather de ŷ  →  Θ(n_test) enteros (etiquetas). Trivial.
  • TRAIN: gather de candidatos → Θ(p·n_test·k) dobles+enteros (distancias+etiq.)
           + fusión O(n_test·p·k) en la raíz. Crece con p, n_test y k.

El trabajo de cómputo es IDÉNTICO (n_test·n_train·3d en ambas); lo que cambia es
el overhead. Este script mide t_compute, t_bcast(X_test), t_scatter(X_train),
t_gather(candidatos) y t_merge por separado, para cuantificar ese overhead.

La accuracy es idéntica al secuencial: la unión de los k-menores locales de cada
proceso contiene siempre a los k-menores globales (un vecino del top-k global está
en el top-k de su propio fragmento), por lo que la fusión reconstruye el resultado
exacto.

Uso:
    DATA_SIZE=5000 IT=1 mpirun -n 4 python src/knn_train_parallel.py
"""
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import csv, os

CSV_PATH = "src/results_train.csv"
HEADERS  = [
    "paradigm", "stage", "it", "n", "p",
    "n_train", "n_test", "k",
    "t_total", "t_compute", "t_comm",
    "t_bcast", "t_scatter", "t_gather", "t_merge",
    "accuracy", "flops", "flops_per_sec",
]
N_FEAT = 64
K      = 3
IT        = int(os.environ.get("IT", 1))
DATA_SIZE = int(os.environ.get("DATA_SIZE", 1797))
DATA_DIR  = "src/data"
FLOP_PER_DIST = 3 * N_FEAT
BATCH_TARGET_BYTES = 256 * 1024 * 1024


def ensure_csv():
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HEADERS)


def append_row(row):
    ensure_csv()
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([row[h] for h in HEADERS])


def knn_local_candidates(X_test, Xtr_local, ytr_local, k):
    """k-menores LOCALES (distancia², etiqueta) por punto de test, por lotes.
    Si el fragmento local tiene < k puntos, rellena con inf / -1."""
    n_test  = X_test.shape[0]
    n_local = Xtr_local.shape[0]
    cand_d  = np.full((n_test, k), np.inf, dtype=np.float64)
    cand_l  = np.full((n_test, k), -1,     dtype=np.int32)
    if n_local == 0:
        return cand_d, cand_l
    kk = min(k, n_local)
    per_row = n_local * N_FEAT * 8
    batch = int(np.clip(BATCH_TARGET_BYTES // max(per_row, 1), 8, 1024))
    for s in range(0, n_test, batch):
        e = min(s + batch, n_test)
        diff = X_test[s:e, None, :] - Xtr_local[None, :, :]
        D = (diff ** 2).sum(axis=2)               # distancia² (b, n_local)
        idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
        rows = np.arange(e - s)[:, None]
        cand_d[s:e, :kk] = D[rows, idx]
        cand_l[s:e, :kk] = ytr_local[idx]
    return cand_d, cand_l


def majority_vote_rows(labels_2d):
    """Voto mayoritario por fila sobre (n_test, m) etiquetas (ignora -1)."""
    n = labels_2d.shape[0]
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        row = labels_2d[i]
        row = row[row >= 0]
        out[i] = int(np.bincount(row, minlength=10).argmax())
    return out


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
            digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target)
        X_train = X_train.astype(np.float64); X_test = X_test.astype(np.float64)
        y_train = y_train.astype(np.int32);   y_test = y_test.astype(np.int32)
    n_train, n_test = int(X_train.shape[0]), int(X_test.shape[0])
else:
    X_train = y_train = X_test = y_test = None
    n_train = n_test = 0

# ── Paso 2: bcast de metadatos + de X_test (¡TODOS necesitan todo el test!) ──
_t0 = MPI.Wtime()
meta = comm.bcast({"n_train": n_train, "n_test": n_test} if rank == 0 else None, root=0)
n_train, n_test = meta["n_train"], meta["n_test"]
if rank != 0:
    X_test = np.empty((n_test, N_FEAT), dtype=np.float64)
comm.Bcast(X_test, root=0)
t_bcast = MPI.Wtime() - _t0

# ── Paso 3: scatter del TRAIN (cada proceso recibe un fragmento disjunto) ────
counts = np.array([n_train // size + (1 if i < n_train % size else 0)
                   for i in range(size)], dtype=np.int32)
local_ntr = int(counts[rank])
displ = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int32)

_t0 = MPI.Wtime()
if rank == 0:
    Xtr_flat = np.ascontiguousarray(X_train.ravel(), dtype=np.float64)
    send_X = [Xtr_flat, (counts * N_FEAT), (displ * N_FEAT), MPI.DOUBLE]
    send_y = [np.ascontiguousarray(y_train), counts, displ, MPI.INT]
else:
    send_X = send_y = None
loc_Xflat = np.empty(local_ntr * N_FEAT, dtype=np.float64)
loc_y     = np.empty(local_ntr, dtype=np.int32)
comm.Scatterv(send_X, [loc_Xflat, local_ntr * N_FEAT, MPI.DOUBLE], root=0)
comm.Scatterv(send_y, [loc_y, local_ntr, MPI.INT], root=0)
Xtr_local = loc_Xflat.reshape((local_ntr, N_FEAT))
t_scatter = MPI.Wtime() - _t0

# ── Paso 4: cómputo local → k-candidatos PARCIALES por punto de test ─────────
_t0 = MPI.Wtime()
cand_d, cand_l = knn_local_candidates(X_test, Xtr_local, loc_y, K)
t_compute = MPI.Wtime() - _t0

# ── Paso 5: gather de los p·k candidatos por punto (la REDUCCIÓN global) ─────
_t0 = MPI.Wtime()
send_d = np.ascontiguousarray(cand_d.ravel())
send_l = np.ascontiguousarray(cand_l.ravel())
if rank == 0:
    recv_d = np.empty(size * n_test * K, dtype=np.float64)
    recv_l = np.empty(size * n_test * K, dtype=np.int32)
else:
    recv_d = recv_l = None
comm.Gather(send_d, recv_d, root=0)
comm.Gather(send_l, recv_l, root=0)
t_gather = MPI.Wtime() - _t0

# ── Paso 6: fusión en la raíz → k-menores GLOBALES + voto ────────────────────
_t0 = MPI.Wtime()
if rank == 0:
    # (p, n_test, k) → (n_test, p·k)
    alld = recv_d.reshape(size, n_test, K).transpose(1, 0, 2).reshape(n_test, size * K)
    alll = recv_l.reshape(size, n_test, K).transpose(1, 0, 2).reshape(n_test, size * K)
    kk = min(K, size * K)
    idx = np.argpartition(alld, kk - 1, axis=1)[:, :kk]
    rows = np.arange(n_test)[:, None]
    final_l = alll[rows, idx]
    y_pred = majority_vote_rows(final_l)
t_merge = MPI.Wtime() - _t0

t_comm  = t_bcast + t_scatter + t_gather
t_total = t_compute + t_comm + t_merge

# ── Evaluación y guardado (solo raíz) ────────────────────────────────────────
if rank == 0:
    acc   = float(np.mean(y_pred == y_test))
    flops = int(n_test) * int(n_train) * FLOP_PER_DIST
    fps   = flops / t_compute if t_compute > 0 else 0.0
    print(f"[MPI/train] it={IT} n={n_train + n_test} p={size} acc={acc:.4f} "
          f"t_total={t_total:.4f}s t_comp={t_compute:.4f}s t_comm={t_comm:.4f}s "
          f"t_merge={t_merge:.4f}s GFLOP/s={fps/1e9:.4f}")
    append_row({
        "paradigm": "mpi", "stage": "train", "it": IT, "n": n_train + n_test, "p": size,
        "n_train": n_train, "n_test": n_test, "k": K,
        "t_total": round(t_total, 6), "t_compute": round(t_compute, 6),
        "t_comm": round(t_comm, 6), "t_bcast": round(t_bcast, 6),
        "t_scatter": round(t_scatter, 6), "t_gather": round(t_gather, 6),
        "t_merge": round(t_merge, 6),
        "accuracy": round(acc, 4), "flops": flops, "flops_per_sec": round(fps, 2),
    })
