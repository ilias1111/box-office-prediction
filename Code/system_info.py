import keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform
import sys
import torch

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print(f"SciPy {sp.__version__}")
gpu = len(tf.config.list_physical_devices("GPU")) > 0
# print out physical devices
print("Physical devices:", tf.config.list_physical_devices())
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# this ensures that the current MacOS version is at least 12.3+
print(f"MacOS 12.3+  : {torch.backends.mps.is_available()}")
# this ensures that the current current PyTorch installation was built with MPS activated.
print(f"PyTorch MPS : {torch.backends.mps.is_built()}")
