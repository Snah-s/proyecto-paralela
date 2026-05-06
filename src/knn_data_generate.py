"""
knn_data_generate.py — Genera datasets para experimentos KNN-MPI.
============================================================
Para realizar los experimentos necesitamos variar el # de datos, para ello
replicamos con perturbación gaussiana ligera para que las muestras no sean 
idénticas (lo cual enriquece la diversidad del dataset sin afectar la 
naturaleza del cómputo que queremos medir).

Diferencias respecto al enfoque base:
  - Se aplica ruido progresivo por ronda de réplica (cada pasada suma
    ligeramente más varianza), evitando que réplicas de distintas rondas
    sean estadísticamente indistinguibles.
  - Los píxeles se clampean a [0, 16] tras cada perturbación para mantener
    el rango válido del dataset original.
  - Se guarda un .npz por tamaño en ./data/ para no regenerar en cada run.
  - Sanity-checks automáticos: forma, dtype, rango, clases únicas.

FIX: el dataset original (n=1797 o None) ahora también se guarda en caché,
     pues el sh verifica la existencia de digits_n1797.npz para saber si
     ya se generaron los datos. La versión anterior retornaba sin guardar.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import os

SCALE_TARGETS = [5_000, 10_000, 15_000, 20_000]
DATA_DIR      = "src/data"
N_ORIG        = 1797   # tamaño del dataset original de sklearn


def load_scaled_digits(
    n_samples_target: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    noise_std: float = 0.05,
    noise_growth: float = 0.05,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Replica el dataset de dígitos con ruido gaussiano progresivo.

    Parameters
    ----------
    n_samples_target : int or None
        Si es None o <= 1797, devuelve el dataset original.
        Si > 1797, lo replica hasta alcanzar el tamaño deseado.
    test_size : float
        Proporción para test split.
    random_state : int
        Semilla.
    noise_std : float
        std base del ruido gaussiano en las réplicas.
    noise_growth : float
        Incremento de std por ronda de réplica.
    use_cache : bool
        Guarda/carga en ./src/data/digits_n{N}.npz.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    digits = load_digits()
    X_orig = digits.data.astype(np.float64)
    y_orig = digits.target.astype(np.int64)

    # Normalizar n: None → 1797
    n_target = N_ORIG if n_samples_target is None else int(n_samples_target)
    cache_path = os.path.join(DATA_DIR, f"digits_n{n_target}.npz")

    # ── Intentar cargar desde caché ───────────────────────────────
    if use_cache and os.path.isfile(cache_path):
        npz = np.load(cache_path)
        return npz["X_train"], npz["X_test"], npz["y_train"], npz["y_test"]

    # ── Dataset original (sin réplicas) ──────────────────────────
    if n_target <= N_ORIG:
        X_train, X_test, y_train, y_test = train_test_split(
            X_orig, y_orig,
            test_size=test_size,
            random_state=random_state,
            stratify=y_orig,
        )

    # ── Réplicas con ruido progresivo ────────────────────────────
    else:
        rng      = np.random.default_rng(random_state)
        n_rondas = int(np.ceil(n_target / N_ORIG))

        X_parts = [X_orig.copy()]   # ronda 0: originales intactos
        y_parts = [y_orig.copy()]

        for ronda in range(1, n_rondas):
            std_ronda = noise_std + ronda * noise_growth
            noise     = rng.normal(0.0, std_ronda, X_orig.shape)
            X_noisy   = np.clip(X_orig + noise, 0.0, 16.0)
            X_parts.append(X_noisy)
            y_parts.append(y_orig.copy())

        X_full = np.vstack(X_parts)
        y_full = np.concatenate(y_parts)

        # Mezclar y recortar al tamaño exacto
        idx    = rng.permutation(len(X_full))
        X_full = X_full[idx[:n_target]]
        y_full = y_full[idx[:n_target]]

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size=test_size,
            random_state=random_state,
            stratify=y_full,
        )

    # ── Guardar en caché (incluye el original n=1797) ────────────
    if use_cache:
        np.savez_compressed(
            cache_path,
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
        )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────
#  Pregenerar todos los tamaños y sanity-check
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    targets = [N_ORIG] + SCALE_TARGETS

    print(f"{'n_target':>10} │ {'X_train':>14} │ {'X_test':>14} │ "
          f"{'dtype':>8} │ {'rango X':>14} │ {'clases':>6} │ {'archivo':>30}")
    print("─" * 100)

    for n in targets:
        X_tr, X_te, y_tr, y_te = load_scaled_digits(n)

        x_min   = X_tr.min()
        x_max   = X_tr.max()
        n_class = len(np.unique(y_tr))
        archivo = f"digits_n{n}.npz"

        print(f"{n:>10} │ {str(X_tr.shape):>14} │ {str(X_te.shape):>14} │ "
              f"{str(X_tr.dtype):>8} │ [{x_min:5.2f}, {x_max:5.2f}] │ "
              f"{n_class:>6} │ {archivo:>30}")

        # Sanity checks
        assert X_tr.dtype  == np.float64, f"[{n}] dtype X debe ser float64"
        assert y_tr.dtype  == np.int64,   f"[{n}] dtype y debe ser int64"
        assert x_min       >= 0.0,        f"[{n}] píxel mínimo < 0"
        assert x_max       <= 16.0,       f"[{n}] píxel máximo > 16"
        assert n_class     == 10,         f"[{n}] faltan clases ({n_class}/10)"

        cache = os.path.join(DATA_DIR, f"digits_n{n}.npz")
        assert os.path.isfile(cache),     f"[{n}] .npz no fue guardado en {cache}"

    print(f"\n✓ Todos los sanity-checks pasaron.")
    print(f"✓ Datasets guardados en ./{DATA_DIR}/")
    for n in targets:
        path = os.path.join(DATA_DIR, f"digits_n{n}.npz")
        mb   = os.path.getsize(path) / 1e6
        print(f"  {path}  ({mb:.2f} MB)")