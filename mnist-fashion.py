import numpy as np
from torchvision import datasets
from pathlib import Path
import os
import time

"""
# Extreme Learning Machine (ELM) 
"If we keep increasing $n$, do we get 100%?"
Answer: "No, because $W_1$ is random. Eventually, we just start memorizing noise (overfitting). To get higher, we'd need to 'cool' $W_1$ too, which brings us back to backpropagation—the very thing VoodooNet avoids for the sake of speed."
"""
# cache dir
root        = str(Path.home()) + os.sep + 'datasets' 
slice_size  = 20000 # -1 #20000 #10000
n_hidden    = 4000 # Accuracy: 97.15% #1000 Accuracy: 94%

# 1. Load Data
ds_train    = datasets.FashionMNIST(root=root, train=True, download=True)
ds_test     = datasets.FashionMNIST(root=root, train=False)

X = ds_train.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
Y = np.eye(10)[ds_train.targets.numpy()] # One-hot labels

# slice
if slice_size > 0:
    X = X[:slice_size, :]
    Y = Y[:slice_size, :]

# 2. THE GALACTIC LAYER 1: Random Projection (1000 hidden neurons)
# We don't train this. We just project the data into a higher-dim space.

# Uniform 86% normal: 94%
W1 = np.random.randn(784, n_hidden)
b1 = np.random.randn(n_hidden)

print(f'FASHION Galactic 2-Layer Shapes X:{X.shape} Y:{Y.shape} W1: {W1.shape} B1:{b1.shape} H:{n_hidden} Samples:{slice_size}')

# Pass data through the non-linear "gate" (ReLU)
t0  = time.time()
H   = np.maximum(0, X @ W1 + b1) 
e0  = time.time() - t0

# 3. THE MAGIC HAT LAYER 2: Solve for the Ground State - Zero "Training"
t0  = time.time()

# Tikhonov Regularization
# Add a small identity matrix to stabilize and "cool" the weights
#W2_galactic = np.linalg.inv(H.T @ H + 1e-3 * np.eye(n_hidden)) @ H.T @ Y
W2_galactic = np.linalg.pinv(H) @ Y

e1  = time.time() - t0

print(f'Solution Shapes H:{H.shape} W2: {W2_galactic.shape} Elapsed Linear: {e0:.1f}s PInv:{e1:.1f}s')

# 4. Instant Inference
X_test = ds_test.data.numpy().reshape(-1, 28*28) / 255.0

print (f'Inference. Shapes X-test: {X_test.shape}')
 
t0       = time.time()
H_test   = np.maximum(0, X_test @ W1 + b1)
accuracy = np.mean(np.argmax(H_test @ W2_galactic, axis=1) == ds_test.targets.numpy())
e0       = time.time() - t0

print(f"FASHION Galactic 2-Layer Accuracy: {accuracy * 100:.2f}% elpased:{e0:.2f}s")
