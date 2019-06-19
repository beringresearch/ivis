"""
Applying ivis to the MNIST dataset
==================================

Ivis can be applied easily applied to unstructured datasets, including images.
Here we visualise the MNSIT digits dataset using two-dimensional ivis
embeddings.
"""

import os
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from ivis import Ivis

mnist = fetch_openml('mnist_784', version=1)

ivis = Ivis(model='maaten', verbose=0)
embeddings = ivis.fit_transform(mnist.data)

color = mnist.target.astype(int)

plt.figure(figsize=(8, 8), dpi=150)
plt.scatter(x=embeddings[:, 0],
            y=embeddings[:, 1], c=color, cmap="Spectral", s=0.1)
plt.xlabel('ivis 1')
plt.ylabel('ivis 2')
plt.show()

os.remove('annoy.index')
