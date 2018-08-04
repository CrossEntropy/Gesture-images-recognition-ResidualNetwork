import numpy as np

np.random.seed(1)
a = np.random.randn(100, 32, 32, 3)
print(a[:, :, :, 0].mean())
