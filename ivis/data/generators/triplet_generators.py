"""
Triplet generators.

Functions for creating generators that will yield batches of triplets.

Triplets will be created using neighbour matrices, which can either be precomputed or
dynamically generated.
"""

from abc import ABC, abstractmethod
from functools import partial
import itertools
import random
import math
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.sparse import issparse

from ..data import get_uint_ctype

def generator_from_neighbour_matrix(X, Y, neighbour_matrix, batch_size):
    if Y is None:
        return UnsupervisedTripletGenerator(X, neighbour_matrix, batch_size=batch_size)

    return SupervisedTripletGenerator(X, Y, neighbour_matrix, batch_size=batch_size)


class TripletGenerator(Sequence, ABC):
    def __init__(self, X, neighbour_matrix, batch_size=32):
        if batch_size > X.shape[0]:
            raise ValueError('''batch_size value larger than num_rows in dataset
                             (batch_size={}, rows={}). Lower batch_size to a
                             smaller value.'''.format(batch_size, X.shape[0]))
        self.X = X
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size
        self.batched_data = hasattr(X, 'get_batch')
        self.batched_neighbours = hasattr(neighbour_matrix, 'get_batch')
        self._idx_dtype = get_uint_ctype(self.X.shape[0])

    def __len__(self):
        return int(math.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        """Gets one batch of triplets"""
        batch_indices = range(idx * self.batch_size,
                              min((idx + 1) * self.batch_size, self.X.shape[0]))
        # indices shape: (3, batch_size)
        triplet_indices = self.get_triplet_indices(batch_indices)
        label_batch = self.get_labels(batch_indices)

        # Retrieve actual data using triplet_indices
        # triplet_batch shape: (3, batch_size, *X.shape[1:])
        triplet_batch = self.get_triplet_data(triplet_indices)

        return self.output_triplets(triplet_batch, label_batch)

    def get_triplet_indices(self, anchor_indices):
        """Generates and returns the triplet indices corresponding to the provided indexes.
        Neighbours are randomly sampled for each row from self.neighbour_matrix and negative
        samples that are not in the neighbour list are generated randomly."""

        neighbour_cands = self.get_all_neighbour_indices(anchor_indices)
        anchor_indices = np.fromiter(anchor_indices,
                                     dtype=self._idx_dtype,
                                     count=len(anchor_indices))
        try:
            # Auto ragged array creation deprecated in NumPy 1.19, 2019-11-01, will throw error
            neighbour_cands = np.asarray(neighbour_cands, dtype=self._idx_dtype)
        except ValueError:
            # Handle ragged array with slower, more generic method
            return self.get_triplet_indices_generic(anchor_indices, neighbour_cands)

        # Older numpy versions will autocast to dtype=object instead of raise exception
        # Detect ragged array in this case and handle with generic method
        if neighbour_cands.ndim < 2:
            return self.get_triplet_indices_generic(anchor_indices, neighbour_cands)

        # Non-ragged array - use shape info to randomly select indices all at once
        neighbour_indices = neighbour_cands[:, np.random.randint(neighbour_cands.shape[1])]
        negative_indices = self.gen_negative_indices(neighbour_cands)
        return (anchor_indices, neighbour_indices, negative_indices)

    def get_all_neighbour_indices(self, anchor_indices):
        """Retrieves neighbours for the indexes provided from inner neighbour matrix.
        Uses specialized `get_batch` retrieval method if generator is in batched_neighbours mode.
        """
        if self.batched_neighbours:
            return self.neighbour_matrix.get_batch(anchor_indices)
        return [self.neighbour_matrix[idx] for idx in anchor_indices]

    def gen_negative_indices(self, neighbour_matrix):
        """Generate random candidate negative indices until the candidate for every
        row is not present in corresponding row of neighbour_matrix."""
        neighbour_matrix = np.asarray(neighbour_matrix)
        generate_cands = partial(np.random.randint, self.X.shape[0], dtype=self._idx_dtype)
        cands = generate_cands(size=len(neighbour_matrix))

        # Where random cand is present in neighbour row, invalid cand
        invalid_cands = (cands[:, np.newaxis] == neighbour_matrix).any(axis=1)
        n_invalid = invalid_cands.sum()
        while n_invalid > 0:
            cands[invalid_cands] = generate_cands(size=n_invalid)
            invalid_cands = (cands[:, np.newaxis] == neighbour_matrix).any(axis=1)
            n_invalid = invalid_cands.sum()
        return cands

    def get_triplet_data(self, triplet_indices):
        """Maps triplet indices to the actual data in internal data store.
        Returns a numpy array of shape (3, batch_size, *self.X.shape[0])."""
        if self.batched_data:
            # Flatten triplets, get batch of data, then reshape back into triplets
            indices = list(itertools.chain.from_iterable(triplet_indices))
            data = self.X.get_batch(indices)
            triplet_batch = list(zip(*[iter(data)] * 3))
        else:
            if isinstance(self.X, np.ndarray):
                # Fancy index for speed if data is a numpy array
                triplet_indices = np.asarray(triplet_indices, dtype=self._idx_dtype)
                triplet_batch = self.X[triplet_indices]
            else:
                triplet_batch = [[self.X[idx] for idx in seq] for seq in triplet_indices]

        if issparse(self.X):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.asarray(triplet_batch)
        return triplet_batch

    def get_triplet_indices_generic(self, anchor_indices, neighbour_cands=None):
        """Slower, generic way of generating triplet indices that works on
        sequences, not just numpy arrays."""
        if neighbour_cands is None:
            neighbour_cands = self.get_all_neighbour_indices(anchor_indices)

        neighbour_indices = list(map(random.choice, neighbour_cands))
        negative_indices = self.gen_negative_indices_generic(neighbour_cands)
        return (anchor_indices, neighbour_indices, negative_indices)

    def gen_negative_indices_generic(self, neighbour_map):
        """Slower, generic way of generating negative indices that works on
        sequences, not just numpy arrays."""
        cands = [random.randrange(0, self.X.shape[0]) for _ in range(len(neighbour_map))]
        for i in range(len(cands)):
            while cands[i] in neighbour_map[i]:
                cands[i] = random.randrange(0, self.X.shape[0])
        return cands

    @abstractmethod
    def get_labels(self, batch_indices):
        raise NotImplementedError("Override this method with a concrete implementation")

    @abstractmethod
    def output_triplets(self, triplet_batch, label_batch):
        raise NotImplementedError("Override this method with a concrete implementation")


class UnsupervisedTripletGenerator(TripletGenerator):
    def __init__(self, X, neighbour_matrix, batch_size=32):
        super().__init__(X, neighbour_matrix, batch_size)
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)

    def get_labels(self, batch_indices):
        return self.placeholder_labels[:len(batch_indices)]

    def output_triplets(self, triplet_batch, label_batch):
        return tuple([*triplet_batch]), label_batch


class SupervisedTripletGenerator(TripletGenerator):
    def __init__(self, X, Y, neighbour_matrix, batch_size=32):
        super().__init__(X, neighbour_matrix, batch_size)
        self.Y = Y

    def get_labels(self, batch_indices):
        return self.Y[batch_indices]

    def output_triplets(self, triplet_batch, label_batch):
        label_batch = np.asarray(label_batch)
        return tuple([*triplet_batch]), tuple([label_batch] * 2)
