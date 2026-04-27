from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
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


# Total time starts
comm.Barrier()
total_start = MPI.Wtime()


# Root loads dataset
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
  k = None
  X_chunks = None
  y_chunks = None


# Broadcast shared data
comm.Barrier()
bcast_start = MPI.Wtime()

X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
k = comm.bcast(k, root=0)

comm.Barrier()
bcast_end = MPI.Wtime()
local_bcast_time = bcast_end - bcast_start


# Scatter test data
comm.Barrier()
scatter_start = MPI.Wtime()

local_X = comm.scatter(X_chunks, root=0)
local_y = comm.scatter(y_chunks, root=0)

comm.Barrier()
scatter_end = MPI.Wtime()
local_scatter_time = scatter_end - scatter_start


# Local computation
comm.Barrier()
compute_start = MPI.Wtime()

local_pred = [knn_predict(x, X_train, y_train, k) for x in local_X]

comm.Barrier()
compute_end = MPI.Wtime()
local_compute_time = compute_end - compute_start


# Gather predictions
comm.Barrier()
gather_start = MPI.Wtime()

all_preds = comm.gather(local_pred, root=0)
all_true = comm.gather(local_y, root=0)

comm.Barrier()
gather_end = MPI.Wtime()
local_gather_time = gather_end - gather_start


# Total time ends
comm.Barrier()
total_end = MPI.Wtime()
local_total_time = total_end - total_start


# Gather timing metrics
bcast_times = comm.gather(local_bcast_time, root=0)
scatter_times = comm.gather(local_scatter_time, root=0)
compute_times = comm.gather(local_compute_time, root=0)
gather_times = comm.gather(local_gather_time, root=0)
total_times = comm.gather(local_total_time, root=0)

if rank == 0:
  y_pred = np.concatenate(all_preds)
  y_true = np.concatenate(all_true)

  accuracy = np.mean(y_pred == y_true)

  bcast_time = max(bcast_times)
  scatter_time = max(scatter_times)
  compute_time = max(compute_times)
  gather_time = max(gather_times)
  total_time = max(total_times)

  communication_time = bcast_time + scatter_time + gather_time

  print("Parallel KNN with MPI - Version 2")
  print(f"Processes:           {size}")
  print(f"Accuracy:            {accuracy:.4f}")
  print(f"Broadcast time:      {bcast_time:.6f} sec")
  print(f"Scatter time:        {scatter_time:.6f} sec")
  print(f"Compute time:        {compute_time:.6f} sec")
  print(f"Gather time:         {gather_time:.6f} sec")
  print(f"Communication time:  {communication_time:.6f} sec")
  print(f"Total time:          {total_time:.6f} sec")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    ax.axis('off')
plt.suptitle("Sample Predictions (Sequential KNN)")
plt.tight_layout()
plt.show()