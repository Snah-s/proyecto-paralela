# Documentación Técnica del Proyecto

## Propósito del Proyecto

Transformaremos una implementación secuencial del algoritmo **K-Nearest Neighbors (KNN)** en una versión paralela utilizando **MPI (Message Passing Interface)** en Python mediante `mpi4py`.

No se busca únicamente clasificar datos, sino **analizar rendimiento computacional**, eficiencia paralela y escalabilidad.

Este trabajo pertenece al área de:

* Computación Paralela
* High Performance Computing (HPC)
* Optimización de algoritmos
* Benchmarking científico

## Contexto

Se parte del dataset `load_digits()` de `scikit-learn`, el cual contiene imágenes de dígitos manuscritos de tamaño 8x8.

Cada imagen se representa como un vector de dimensión $d = 64$

El algoritmo debe predecir la clase de nuevas imágenes usando KNN.

### Fundamento del Algoritmo KNN

KNN clasifica un punto de prueba comparándolo con todos los datos de entrenamiento.

Para cada muestra de prueba:

1. Se calcula distancia a todos los puntos de entrenamiento.
2. Se ordenan distancias.
3. Se toman los `k` vecinos más cercanos.
4. Se aplica votación mayoritaria.

### Distancia Euclidiana

La métrica usada es:

$$
dist(x,y)=\sqrt{\sum_{i=1}^{d}(x_i-y_i)^2}
$$

Esta operación domina el costo computacional.

### Complejidad Secuencial

Si:

* $n_{train}$: muestras de entrenamiento
* $n_{test}$: muestras de prueba
* $d$: dimensión

Entonces:

$$
T_{seq}=O(n_{test}\cdot n_{train}\cdot d)
$$

KNN no entrena modelo; el costo está en inferencia.

### Paralelización

Cada muestra de prueba puede clasificarse independientemente. Eso permite paralelismo natural:

$$
X_{test} = X_1 \cup X_2 \cup ... \cup X_p
$$

Cada proceso trabaja sobre un subconjunto distinto. Esto reduce tiempo de cómputo idealmente a:

$$
T_p \approx \frac{T_1}{p}
$$

si el costo de comunicación es pequeño.

### Modelo PRAM Conceptual

El problema puede modelarse bajo PRAM como múltiples procesadores que acceden a memoria compartida lógica.

Cada procesador:

* recibe parte de `X_test`
* usa el mismo `X_train`
* calcula predicciones locales

Luego se combinan resultados.

MPI implementa esto de forma distribuida.

## Diseño MPI del Proyecto

### Proceso Root (rank 0)

Responsable de:

* cargar dataset
* dividir train/test
* fragmentar `X_test`
* repartir trabajo
* recolectar resultados
* calcular métricas

### Procesos Worker

Responsables de:

* recibir datos
* ejecutar KNN local
* retornar predicciones

### Operaciones de MPI Utilizadas

`bcast()`: Envía desde root a todos:

* `X_train`
* `y_train`
* `k`

`scatter()`: Distribuye subconjuntos de prueba:

* `X_test`
* `y_test`

`gather()`: Recolecta:

* predicciones
* tiempos locales


## Todo

### Fase Secuencial

#### Objetivo

Medir referencia inicial.

#### Acciones

* ejecutar código original
* medir tiempo total
* medir accuracy

#### Resultado esperado

$$
T_1
$$

Será base para speedup.


### Paralelización Básica

#### Objetivo

Distribuir trabajo usando MPI.

#### Acciones

* root divide `X_test`
* broadcast entrenamiento
* scatter test
* predicción local
* gather resultados

#### Resultado esperado

Versión funcional paralela.


### Instrumentación de Tiempos

Separar costos:

$$
T_{total}=T_{comm}+T_{compute}
$$

Medir:

* broadcast time
* scatter time
* compute time
* gather time

### Benchmarks

Ejecutar con:

$$
p = 1,2,4,8,...
$$

Registrar:

* tiempo total
* tiempo cómputo
* tiempo comunicación
* accuracy

#### Speedup

$$
S(p)=\frac{T_1}{T_p}
$$

Interpretación:

* (S(p)=p): ideal
* (S(p)<p): overhead
* (S(p)>p): raro/superlinear

#### Eficiencia

$$
E(p)=\frac{S(p)}{p}
$$

Mide aprovechamiento de procesos.

#### FLOPS

Estimar operaciones de distancia euclidiana.

Por comparación:

* restas: (d)
* multiplicaciones: (d)
* sumas: (d-1)
* raíz: 1

Aprox:

$$
3d
$$

Total:

$$
FLOP \approx n_{test}\cdot n_{train}\cdot 3d
$$

Rendimiento:

$$
FLOPS/sec = \frac{FLOP}{T_p}
$$

## Escalabilidad

#### Strong Scaling

Problema fijo, aumentan procesos.

#### Weak Scaling

Crece problema junto con procesos.

Analizar cuándo comunicación domina.