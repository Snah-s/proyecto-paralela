# KNN Paralelo con MPI y OpenMP — Clasificación de Dígitos

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MPI](https://img.shields.io/badge/Open%20MPI-4.1-orange?logo=linux)
![mpi4py](https://img.shields.io/badge/mpi4py-3.1-green)
![numba](https://img.shields.io/badge/numba-0.66-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Paralelización del algoritmo K-Nearest Neighbors (KNN) con MPI (memoria distribuida) y numba/OpenMP (memoria compartida) sobre el dataset `load_digits` de scikit-learn.**
Proyecto del curso *Computación Paralela y Distribuida* — UTEC

| Alumno | Participación |
|---|---|
| Ricardo Amiel Acuña Villogas | 100% |
| Camilo Ernesto Soto Cristobal | 100% |

</div>

---

## Tabla de Contenidos

- [Descripción](#-descripción)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos e Instalación](#-requisitos-e-instalación)
- [Uso](#-uso)
- [Diseño: una implementación, tres etapas](#-diseño-una-implementación-tres-etapas)
- [Resultados](#-resultados)
- [Documentación](#-documentación)

---

## 📖 Descripción

Se transforma la versión secuencial de KNN (`src/knn_digits_sec.py`) en implementaciones
paralelas, evaluando **speedup, escalabilidad, tiempos de comunicación, FLOP/s y precisión**.
El diseño sigue el paradigma **PRAM CREW** (Concurrent Read, Exclusive Write): todos los
procesos leen `X_train` (solo lectura), y cada proceso escribe su partición disjunta de `y_pred`.

```
Raíz carga datos
     │
     ├── bcast(X_train, y_train) ──→ todos los procesos   (Lectura Concurrente)
     │
     ├── scatter(X_test) ──→ fracción local a cada proceso
     │                         │
     │              ┌──────────┴──────────┐
     │           P0: KNN      P1: KNN  ...  Pk: KNN        (Escritura Exclusiva)
     │              └──────────┬──────────┘
     └── gather(y_pred) ←──────┘
```

**Se paraleliza la fase de TEST, no la de TRAIN** — KNN es un *lazy learner* y todo el
cómputo está en la inferencia.

Además de MPI, se implementa una versión de **memoria compartida** (numba `prange` → hilos
OpenMP reales) para comparar paradigmas empíricamente.

---

## 🗂 Estructura del Proyecto

```
proyecto-paralela/
├── src/
│   ├── knn_digits_sec.py         # Secuencial base (referencia)
│   ├── knn_data_generate.py      # Genera datasets escalados 1797–20000 (.npz)
│   ├── knn_paralelo.py           # MPI — 1 código, 3 etapas (STAGE=loop|vec|buf)
│   ├── knn_omp.py                # Memoria compartida (numba prange / OpenMP)
│   ├── make_figures.py           # Genera las figuras vectorizadas del informe
│   ├── experimental_analysis.ipynb  # Notebook de análisis (mismas figuras)
│   ├── run_experiments.sh        # Sweep MPI + OMP (resume + oversubscribe)
│   ├── results_mpi.csv           # Resultados MPI  (generado, 300 filas)
│   ├── results_omp.csv           # Resultados OMP  (generado, 100 filas)
│   ├── data/                     # Datasets .npz (generado)
│   └── partial_project_version/  # Código archivado del proyecto parcial
├── Proyecto_Final_Paralela/      # Informe final IEEE (main.tex, referencias.bib, figures/)
└── README.md
```

---

## ⚙️ Requisitos e Instalación

### Hardware de los experimentos

| Recurso | Valor |
|---|---|
| CPU | Intel i7-1165G7 — **4 núcleos físicos / 8 hilos (Hyper-Threading)** |
| RAM | 16 GB |
| MPI | Open MPI 4.1.6 |

> Este hardware coincide con el escenario "4 procesadores / 8 hilos" del enunciado, lo que
> permite responder empíricamente el efecto del Hyper-Threading en el speedup.

### Software

```
Python >= 3.11 · Open MPI >= 4.1 · mpi4py · numpy · scikit-learn · scipy · numba · pandas · matplotlib
```

### Opción A — conda / micromamba (según `environment.yml`)

```bash
micromamba env create -f environment.yml   # o: conda env create -f environment.yml
micromamba activate knn-mpi                 # o: conda activate knn-mpi
pip install numba                           # (numba se usa en la versión OMP)
```

### Opción B — venv reutilizando el Open MPI del sistema (probado)

```bash
sudo apt install openmpi-bin libopenmpi-dev          # Ubuntu/Debian
python3 -m venv --system-site-packages .venv          # reutiliza mpi4py del sistema si existe
source .venv/bin/activate
pip install scikit-learn pandas matplotlib numba
```

> ⚠️ `mpi4py` debe estar compilado con el mismo MPI que usa `mpirun`. Verifica:
> ```bash
> python -c "from mpi4py import MPI; print(MPI.Get_library_version()[:50])"
> mpirun --version
> ```

---

## 🚀 Uso

### 1. Generar los datasets

```bash
python src/knn_data_generate.py     # crea src/data/digits_n{1797,5000,10000,15000,20000}.npz
```

### 2. Ejecutar una etapa manualmente

```bash
# MPI, etapa 'buf', n=5000, p=4
STAGE=buf DATA_SIZE=5000 IT=1 mpirun -n 4 python src/knn_paralelo.py

# p=8 requiere --oversubscribe (solo hay 4 núcleos físicos)
STAGE=buf DATA_SIZE=20000 IT=1 mpirun --oversubscribe -n 8 python src/knn_paralelo.py

# Memoria compartida (OMP), 4 hilos
THREADS=4 DATA_SIZE=20000 IT=1 python src/knn_omp.py
```

Variables: `STAGE ∈ {loop, vec, buf}`, `DATA_SIZE ∈ {1797,5000,10000,15000,20000}`, `THREADS` (OMP), `IT` (nº de iteración).

### 3. Sweep completo de experimentos

```bash
chmod +x src/run_experiments.sh
./src/run_experiments.sh              # MPI (3 etapas) + OMP, todos los p y n, 5 iters
ONLY=omp ./src/run_experiments.sh     # solo OMP     (o ONLY=mpi)
FORCE=1  ./src/run_experiments.sh     # re-corre todo (ignora resume)
```

Genera `src/results_mpi.csv` (300 filas) y `src/results_omp.csv` (100 filas).
**Resume automático**: al re-ejecutar, omite las combinaciones ya presentes en el CSV.

### 4. Generar figuras / analizar

```bash
python src/make_figures.py                          # → Proyecto_Final_Paralela/figures/*.pdf
# o de forma interactiva:
jupyter notebook src/experimental_analysis.ipynb
```

---

## 🔄 Diseño: una implementación, tres etapas

Un solo programa (`knn_paralelo.py`) con el comportamiento seleccionable por `STAGE`
(registra el desarrollo incremental que pide el enunciado):

| Etapa | Distancia | Comunicación | Nota |
|---|---|---|---|
| **loop** | Por punto con NumPy | `bcast`/`scatter`/`gather` (pickle) | Baseline directo |
| **vec** | Matricial 3-D **por lotes** + `argpartition` | pickle | Acota memoria (evita el tensor de ~32 GB en n=20000) |
| **buf** | Igual a `vec` | `Bcast`/`Scatterv`/`Gatherv` (buffers) | Sin serialización pickle |

**FLOPs por distancia euclidiana:** `3·d = 192` (d restas + d mult. + (d−1) sumas + 1 raíz),
`FLOPs_total = n_test · n_train · 3d`.

**Modelo de comunicación (α–β, árbol binomial):** `T_comm = Θ(log p · (α + β·m))` — el factor
`log p` es el número de rondas de la colectiva, no una constante ni `(p−1)`.

---

## 📊 Resultados

Máquina **4c/8t** · Open MPI 4.1.6 · 5 iteraciones por `(etapa, p, n)` · agregación por **mediana**
(robusta al *thermal throttling* del portátil). `p=8` está **oversuscrito** (8 procesos / 4 núcleos).

### Speedup y eficiencia — etapa `buf`

| n | S(2) | E(2) | S(4) | E(4) | S(8) |
|---|---|---|---|---|---|
| 1,797 | 1.96 | 0.98 | 3.21 | 0.80 | 1.53 |
| 10,000 | 1.94 | 0.97 | 2.88 | 0.72 | 3.35 |
| 15,000 | 2.19 | 1.09 | 3.32 | 0.83 | **3.81** |
| 20,000 | 1.73 | 0.86 | 2.35 | 0.59 | 2.70 |

> El speedup **satura cerca de p=4** (núcleos físicos). Más allá, el Hyper-Threading y la
> oversubscripción aportan poco: KNN es *compute-bound*. Óptimo estimado **p\* ≈ 5.4**
> (modelo `T = a/p + b·p`, con a=25.2 s, b=0.87 s).

### MPI vs OMP — tiempo de cómputo a p=4

| n | MPI `buf` (s) | OMP numba (s) | Aceleración |
|---|---|---|---|
| 1,797 | 0.093 | 0.0024 | **38.8×** |
| 10,000 | 3.072 | 0.0776 | **39.6×** |
| 20,000 | 10.921 | 0.3642 | **30.0×** |

> En un solo nodo, la **memoria compartida (OMP) es 30–40× más rápida** que MPI: evita la
> réplica de `X_train`, la serialización y la latencia `log p`. Pico: **46–67 GFLOP/s** (OMP)
> vs ~1.3 GFLOP/s (MPI). MPI se justifica por su escalado a **múltiples nodos**.

### Precisión (invariante en p)

```
n=1797 → 0.9861 · n=5000 → 0.9960 · n≥10000 → 1.0000     (varía con n por la augmentación, NO con p)
```

La invarianza en `p` valida que la paralelización es matemáticamente equivalente al secuencial.

### Escalabilidad (dos vías)

- **Brent** (sin comunicación): `n ∝ √p`
- **Isoeficiencia** (con comunicación `log p`): `n ∝ p·log p`

---

## 📚 Documentación

| Documento | Contenido |
|---|---|
| `Proyecto_Final_Paralela/main.tex` | Informe final IEEE |
