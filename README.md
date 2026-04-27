# Parallel KNN MPI Digits

Implementación y análisis de rendimiento de K-Nearest Neighbors (KNN) paralelizado con MPI sobre el dataset `load_digits` de scikit-learn.

hola

## Descripción

Este proyecto tiene como objetivo transformar una versión secuencial del algoritmo KNN en una implementación paralela usando `mpi4py`, evaluando rendimiento computacional, speedup, escalabilidad, tiempos de comunicación y precisión del modelo.

Se trabaja sobre el dataset clásico de reconocimiento de dígitos manuscritos incluido en `scikit-learn`.



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


