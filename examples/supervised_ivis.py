"""
Supervised Dimensionality Reduction with ivis
=============================================

ivis is able to make use of any provided class labels to perform supervised
dimensionality reduction. Supervised embeddings combine the distance-based
characteristics of the unsupervised ivis algorithm with clear class boundaries
between the class categories. The resulting embeddings encode relevant
class-specific information into lower dimensional space, making them useful
for enhancing the performance of a classifier.

To train ivis in supervised mode, simply provide the labels to the fit
method’s Y parameter. These labels should be a list of 0-indexed integers with
each integer corresponding to a class.
"""

import numpy as np
from keras.datasets import mnist
from ivis import Ivis

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Rescale to 0-1
X_train = X_train / 255.
X_test = X_test / 255.

# Flatten images to 1D vectors
X_train = np.reshape(X_train, (len(X_train), 28 * 28))
X_test = np.reshape(X_test, (len(X_test), 28 * 28))

model = Ivis(n_epochs_without_progress=5)
model.fit(X_train, Y_train)
