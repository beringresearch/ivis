"""Generators for non-triplet data"""

import tensorflow as tf
import numpy as np

from scipy.sparse import issparse


class KerasSequence(tf.keras.utils.Sequence):
    """Wraps inputs into a Keras Sequence to allow Keras models to predict on
    arbitrary inputs which may be out of memory."""
    def __init__(self, X, batch_size=32):
        self.X = X
        self.batch_size = batch_size
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)
    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))
    def __getitem__(self, index):
        batch_indices = range(index * self.batch_size, min((index + 1) * self.batch_size, self.X.shape[0]))

        if issparse(self.X):
            batch = np.array([self.X[i].toarray() for i in batch_indices])
            batch = np.squeeze(batch)
        else:
            batch = np.array([self.X[i] for i in batch_indices])
        placeholder_labels = self.placeholder_labels[:len(batch_indices)]
        return batch, placeholder_labels
