"""
Triplet generators.

Functions for creating generators that will yield batches of triplets.

Triplets will be created using neighbour matrices, which can either be precomputed or
dynamically generated.
"""

from abc import ABC, abstractmethod
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.sparse import issparse


def generator_from_neighbour_matrix(X, Y, neighbour_matrix, batch_size):
    if Y is None:
        return UnsupervisedTripletGenerator(X, neighbour_matrix, batch_size=batch_size)

    return SupervisedTripletGenerator(X, Y, neighbour_matrix, batch_size=batch_size)


class TripletGenerator(Sequence, ABC):
    def __init__(self, X, neighbour_matrix, batch_size=32):
        if batch_size > X.shape[0]:
            raise Exception('''batch_size value larger than num_rows in dataset
                            (batch_size={}, rows={}). Lower batch_size to a
                            smaller value.'''.format(batch_size, X.shape[0]))
        self.X = X
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size,
                              min((idx + 1) * self.batch_size, self.X.shape[0]))

        label_batch = self.get_labels(batch_indices)
        triplet_batch = [self.get_triplet(row_index)
                         for row_index in batch_indices]

        if issparse(self.X):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)

        return self.output_triplets(triplet_batch, label_batch)

    def get_triplet(self, idx):
        triplet = []
        neighbour_list = self.get_neighbours(idx)
        neighbour_list = np.array(neighbour_list, dtype=np.uint32)

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint, usually faster.
        negative_ind = np.random.randint(0, self.X.shape[0])
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplet += [self.X[idx], self.X[neighbour_ind], self.X[negative_ind]]
        return triplet

    def get_neighbours(self, idx):
        return self.neighbour_matrix[idx]

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
        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), label_batch


class SupervisedTripletGenerator(TripletGenerator):
    def __init__(self, X, Y, neighbour_matrix, batch_size=32):
        super().__init__(X, neighbour_matrix, batch_size)
        self.Y = Y

    def get_labels(self, batch_indices):
        return self.Y[batch_indices]

    def output_triplets(self, triplet_batch, label_batch):
        label_batch = np.array(label_batch)
        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), tuple([label_batch, label_batch])
