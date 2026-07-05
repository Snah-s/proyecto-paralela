from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def euclidean_distance(a, b):
  return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(test_point, X_train, y_train, k):
  distances = [euclidean_distance(test_point, x) for x in X_train]
  k_indices = np.argsort(distances)[:k]
  k_labels = [y_train[i] for i in k_indices]
  return Counter(k_labels).most_common(1)[0][0]


if rank == 0:
  digits = load_digits()

  X_train, X_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42
  )

  k = 3

  X_chunks = np.array_split(X_test, size)
  y_chunks = np.array_split(y_test, size)

else:
  X_train = None
  y_train = None
  y_test = None
  X_chunks = None
  y_chunks = None
  k = None


# Broadcast shared data
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
k = comm.bcast(k, root=0)

# Scatter test chunks
local_X = comm.scatter(X_chunks, root=0)
local_y = comm.scatter(y_chunks, root=0)

comm.Barrier()
start = MPI.Wtime()

# Local predictions
local_pred = [knn_predict(x, X_train, y_train, k) for x in local_X]

comm.Barrier()
end = MPI.Wtime()

local_time = end - start

# Gather
all_preds = comm.gather(local_pred, root=0)
all_true = comm.gather(local_y, root=0)
times = comm.gather(local_time, root=0)

# Root computes metrics
if rank == 0:
  y_pred = np.concatenate(all_preds)
  y_true = np.concatenate(all_true)

  accuracy = np.mean(y_pred == y_true)
  total_time = max(times)

  print(f"Processes: {size}")
  print(f"Accuracy: {accuracy:.4f}")
  print(f"Parallel Time: {total_time:.4f} sec")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    ax.axis('off')
plt.suptitle("Sample Predictions (Sequential KNN)")
plt.tight_layout()
plt.show()