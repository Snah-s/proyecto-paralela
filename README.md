# KNN Paralelo con MPI — Clasificación de Dígitos

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![MPI](https://img.shields.io/badge/Open%20MPI-5.0-orange?logo=linux)
![mpi4py](https://img.shields.io/badge/mpi4py-4.x-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Paralelización del algoritmo K-Nearest Neighbors (KNN) usando MPI para clasificar el dataset `load_digits` de scikit-learn.**  
Proyecto del curso *Computación Paralela y Distribuida* — Facultad de Computación

| Integrante | Departamento |
|---|---|
| Ricardo Amiel Acuña Villogas | Data Science |
| Camilo Ernesto Soto Cristobal | Data Science |

</div>

---

## Tabla de Contenidos

- [Descripción](#-descripción)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Versiones Paralelas](#-versiones-paralelas)
- [Resultados](#-resultados)
- [Análisis](#-análisis)

---

## 📖 Descripción

Este proyecto tiene como objetivo transformar una versión secuencial del algoritmo KNN en una implementación paralela usando `mpi4py`, evaluando rendimiento computacional, speedup, escalabilidad, tiempos de comunicación y precisión del modelo.

Se trabaja sobre el dataset clásico de reconocimiento de dígitos manuscritos incluido en `scikit-learn`.

Se implementaron **3 versiones progresivas** del algoritmo KNN en paralelo con el paradigma **PRAM CREW** (Concurrent Read, Exclusive Write) usando MPI:

```
Raíz carga datos
     │
     ├── bcast(X_train, y_train) ──→ todos los procesos
     │
     ├── scatter(X_test) ──→ fracción local a cada proceso
     │                         │
     │              ┌──────────┴──────────┐
     │           P0: KNN      P1: KNN  ...  Pk: KNN   ← cómputo independiente
     │              └──────────┬──────────┘
     └── gather(y_pred) ←──────┘
```

Cada versión optimiza una dimensión distinta:

| Versión | Distancia | Comunicación | Mejora principal |
|---------|-----------|--------------|-----------------|
| **V1** | Loop Python (`for x in X_train`) | `bcast` + `scatter` + `gather` pickle | Baseline paralelo |
| **V2** | Matricial NumPy broadcasting + `argpartition` | `bcast` + `scatter` + `gather` pickle | Cómputo ×6 más rápido |
| **V3** | Igual a V2 | `Bcast`/`Scatterv`/`Gatherv` buffers MPI | Sin serialización pickle |

---

## Tecnologías utilizadas

- Python 3.11+
- NumPy
- Scikit-learn
- Matplotlib
- Pandas
- MPI / OpenMPI
- mpi4py
- Conda / Micromamba / venv



# Instalación de dependencias

El proyecto soporta tres formas de instalación:

- Micromamba (recomendado)
- Conda
- Entorno virtual estándar (`venv`)



# Opción 1: Micromamba (Recomendado)

## Instalar micromamba

Si no lo tienes instalado:

```bash
curl -Ls https://micro.mamba.pm/install.sh | bash
````

Reinicia terminal o recarga shell.

## Crear entorno

```bash
micromamba env create -f environment.yml
```

## Activar entorno

```bash
micromamba activate knn-mpi
```



# Opción 2: Conda

## Crear entorno

```bash
conda env create -f environment.yml
```

## Activar entorno

```bash
conda activate knn-mpi
```



# Opción 3: venv + pip

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```


# Dependencia adicional para MPI (solo venv)

Si usas `venv`, necesitas instalar MPI manualmente.

## Ubuntu / Debian

```bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
```

## Arch Linux

```bash
sudo pacman -S openmpi
```

## 🗂 Estructura del Proyecto

```
proyecto-paralela/
├── src/
│   ├── knn_data_generate.py        # Genera datasets escalados (5k–20k)
│   ├── knn_paralelo_v1.py          # V1: loop Python + pickle MPI
│   ├── knn_paralelo_v2.py          # V2: NumPy vectorizado + pickle MPI
│   ├── knn_paralelo_v3.py          # V3: NumPy vectorizado + buffer MPI
│   ├── run_experiments.sh          # Script de experimentos (con resume)
│   ├── results_knn_v1.csv          # Resultados V1 (generado)
│   ├── results_knn_v2.csv          # Resultados V2 (generado)
│   ├── results_knn_v3.csv          # Resultados V3 (generado)
│   └── data/                       # Datasets .npz generados
│       ├── digits_n1797.npz
│       ├── digits_n5000.npz
│       ├── digits_n10000.npz
│       ├── digits_n15000.npz
│       └── digits_n20000.npz
├── experimental_analysis_v2.ipynb  # Notebook de análisis y gráficas
└── README.md
```

---

## ⚙️ Requisitos

### Hardware
| Recurso | Mínimo | Usado en experimentos |
|---------|--------|----------------------|
| Núcleos físicos | 2 | **12 núcleos físicos** |
| Hilos lógicos | 4 | **24 hilos (Hyperthreading)** |
| RAM | 4 GB | 32 GB |

### Software
```
Python      >= 3.11
Open MPI    >= 5.0        ← importante: debe ser Open MPI, no MPICH
mpi4py      >= 3.1
numpy       >= 1.24
scikit-learn >= 1.3
scipy       >= 1.10
```

> ⚠️ **Advertencia**: `mpi4py` debe estar compilado con la misma implementación MPI que usas para lanzar (`mpirun`). Si tu sistema tiene MPICH instalado, usa `$CONDA_PREFIX/bin/mpirun` para garantizar la compatibilidad con Open MPI de conda.

---

## 🔧 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/<usuario>/proyecto-paralela.git
cd proyecto-paralela
```

### 2. Crear entorno conda
```bash
conda env create -f environment.yml
conda activate knn-mpi
```

### 3. Verificar que mpirun es Open MPI
```bash
# Debe mostrar "Open MPI", NO "HYDRA"
$CONDA_PREFIX/bin/mpirun --version

# Verificar que mpi4py usa el mismo
python -c "from mpi4py import MPI; print(MPI.Get_library_version()[:50])"
```

---

## Uso

### Paso 1 — Generar los datasets

```bash
python src/knn_data_generate.py
```

Esto genera en `src/data/` los archivos `.npz` para n ∈ {1797, 5000, 10000, 15000, 20000}.  
Cada dataset escala el original con **ruido gaussiano progresivo** por ronda (sin data leakage).

```
 n_target │      X_train │       X_test │    dtype │      rango X │ clases
─────────────────────────────────────────────────────────────────────────
    1797  │  (1437, 64)  │   (360, 64)  │  float64 │ [ 0.00, 16.00] │     10
    5000  │  (4000, 64)  │  (1000, 64)  │  float64 │ [ 0.00, 16.00] │     10
   10000  │  (8000, 64)  │  (2000, 64)  │  float64 │ [ 0.00, 16.00] │     10
   15000  │ (12000, 64)  │  (3000, 64)  │  float64 │ [ 0.00, 16.00] │     10
   20000  │ (16000, 64)  │  (4000, 64)  │  float64 │ [ 0.00, 16.00] │     10
```

### Paso 2 — Ejecutar una versión manualmente

```bash
# V1 con p=4 procesos, dataset n=5000, iteración 1
IT=1 DATA_SIZE=5000 $CONDA_PREFIX/bin/mpirun -n 4 python src/knn_paralelo_v1.py

# V3 con p=8 (requiere --oversubscribe en máquinas de 4 núcleos)
IT=1 DATA_SIZE=20000 $CONDA_PREFIX/bin/mpirun --oversubscribe -n 8 python src/knn_paralelo_v3.py
```

### Paso 3 — Correr experimentos completos

```bash
chmod +x src/run_experiments.sh

# Solo n=1797 (rápido, para prueba)
./src/run_experiments.sh

# Todos los tamaños (n ∈ {1797, 5000, 10000, 15000, 20000})
ALL_SIZES=1 ./src/run_experiments.sh

# Solo una versión
VERSION=v1 ALL_SIZES=1 ./src/run_experiments.sh

# Re-correr todo ignorando resultados previos
FORCE=1 ALL_SIZES=1 ./src/run_experiments.sh
```

> **Resume automático**: si interrumpes con `Ctrl+C`, al re-ejecutar el script detecta qué filas ya están en el CSV y retoma exactamente donde se quedó.

El script genera:
```
src/results_knn_v1.csv   ← 100 filas (5 its × 4 p × 5 n)
src/results_knn_v2.csv   ← 100 filas
src/results_knn_v3.csv   ← 100 filas
```

### Paso 4 — Analizar resultados

```bash
jupyter notebook experimental_analysis_v2.ipynb
```

---

## 🔄 Versiones Paralelas

### FLOPs por distancia euclidiana
```
FLOPs(d=64) = 3·d + 1 = 193  por cada par (x_test, x_train)
FLOPs_total = n_test × n_train × 193
```

### V1 — Baseline paralelo
- **Distancia**: loop Python `for x in X_train`
- **Comunicación**: `comm.bcast` + `comm.scatter` + `comm.gather` (serialización pickle)
- **Complejidad comunicación**: O(n_train · d · p) por el bcast a todos

### V2 — Cómputo vectorizado
- **Distancia**: `D = sqrt(((local_X[:,None,:] - X_train[None,:,:])**2).sum(2))`  
  Una sola operación NumPy: shape `(local_n, n_train)`
- **Votación**: `np.argpartition` O(n) vs argsort O(n log n)
- **Comunicación**: igual que V1

### V3 — Comunicación por buffers MPI
- **Distancia**: igual que V2
- **Comunicación**: `Bcast` (mayúscula) + `Scatterv` + `Gatherv`  
  Transmite buffers NumPy directamente sin pickle → menor overhead para datasets grandes

---

## 📊 Resultados

Experimentos ejecutados en computadora de laboratorio: **12 núcleos físicos / 24 hilos lógicos**, Open MPI 5.0, 5 iteraciones por combinación `(p, n)`.

### Tabla resumen — V1 (n=1797, k=3)

| p | T_total (ms) | T_comm (ms) | T_compute (ms) | Speedup S | Eficiencia E | Granularidad G | GFLOP/s |
|---|---|---|---|---|---|---|---|
| 2⁰=1 | 686.02 | 0.93 | 685.09 | 1.000 | 1.000 | 739.2 | 0.146 |
| 2¹=2 | 349.28 | 1.89 | 347.40 | 1.964 | 0.982 | 184.3 | 0.287 |
| 2²=4 | 179.33 | 3.83 | 175.50 | 3.826 | 0.956 | 45.8 | 0.569 |
| 2³=8 | 94.41 | 4.63 | 89.78 | 7.266 | 0.908 | 19.4 | 1.113 |

### Ley de Amdahl — fracción paralela f

| n | f ajustado | Límite S(∞) | S(p=4) | E(p=4) |
|---|---|---|---|---|
| 1,797 | 0.9855 | 69.1× | 3.825 | 0.956 |
| 5,000 | 0.9897 | 97.2× | 3.796 | 0.949 |
| 10,000 | 0.9886 | 87.6× | 3.768 | 0.942 |
| 15,000 | 0.9823 | 56.4× | 3.696 | 0.924 |
| 20,000 | 0.9851 | 67.3× | 3.759 | 0.940 |

### p* óptimo teórico (modelo T = α/p + β·p)

Para n=20,000: **α = 85.90 s**, **β = 0.176 s** → **p\* = √(α/β) ≈ 22**

> El cómputo domina ampliamente (G > 40 en todos los casos), lo que indica que añadir más procesos es beneficioso hasta p≈22 para el dataset más grande.

### Accuracy (constante en todos los p)
```
Accuracy = 0.9861  (independiente de p — resultado correcto)
```

---

## 📈 Análisis

El notebook `experimental_analysis_v2.ipynb` genera automáticamente:

| Figura | Descripción |
|--------|-------------|
| `fig_speedup_eficiencia.pdf` | Speedup y eficiencia — una línea por n |
| `fig_t_total_vs_p.pdf` | Tiempo total vs p — hue = n |
| `fig_crossover_popt.pdf` | Ajuste T_compute/T_comm/T_total + p* |
| `fig_comm_breakdown.pdf` | T_compute vs T_comm + desglose Bcast/Scatter/Gather |
| `fig_flops_vs_p.pdf` | GFLOP/s vs p — hue = n |
| `fig_escalabilidad.pdf` | t_compute vs n + speedup vs n |
| `fig_amdahl.pdf` | Ley de Amdahl con ajuste por n |
| `tabla_resumen.pdf` | Tabla completa de métricas |
