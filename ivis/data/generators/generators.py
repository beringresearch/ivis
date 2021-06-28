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
        self.batched_data = hasattr(X, 'get_batch')
    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))
    def __getitem__(self, index):
        batch_indices = range(index * self.batch_size, min((index + 1) * self.batch_size, self.X.shape[0]))

        if self.batched_data:
            batch = self.X.get_batch(batch_indices)
        else:
            batch = [self.X[i] for i in batch_indices]

        if issparse(self.X):
            batch = [ele.toarray() for ele in batch]
            batch = np.squeeze(batch)

        placeholder_labels = self.placeholder_labels[:len(batch_indices)]
        return np.asarray(batch), placeholder_labels
