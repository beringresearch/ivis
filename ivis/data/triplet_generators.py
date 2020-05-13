"""
Triplet generators.

Functions for creating generators that will yield batches of triplets.

Triplets will be created using KNNs, which can either be precomputed or
dynamically generated.
- generate_knn_triplets_from_neighbour_list will precompute KNNs
- generate_knn_triplets_from_annoy_index will dynamically generate KNNs

Where possible, precomputed KNNs are advised for speed, but where memory is
a concern, dynamically generated triplets can be useful.

"""

import numpy as np
from .knn import extract_knn
from annoy import AnnoyIndex
from tensorflow.keras.utils import Sequence
from scipy.sparse import issparse
from functools import partial
import tensorflow as tf


def generator_from_knn_matrix(X, Y, neighbour_matrix, k, batch_size, search_k, verbose=1):
    if k >= X.shape[0] - 1:
        raise Exception('''k value greater than or equal to (num_rows - 1)
                        (k={}, rows={}). Lower k to a smaller
                        value.'''.format(k, X.shape[0]))
    if batch_size > X.shape[0]:
        raise Exception('''batch_size value larger than num_rows in dataset
                        (batch_size={}, rows={}). Lower batch_size to a
                        smaller value.'''.format(batch_size, X.shape[0]))
    if Y is None:
        return create_knn_triplet_dataset(X, neighbour_matrix, batch_size=batch_size)
    else:
        return create_labeled_knn_triplet_dataset(X, Y, neighbour_matrix,
                                                  batch_size=batch_size)


def generator_from_index(X, Y, index_path, k, batch_size, search_k=-1,
                         precompute=True, verbose=1):
    if k >= X.shape[0] - 1:
        raise Exception('''k value greater than or equal to (num_rows - 1)
                        (k={}, rows={}). Lower k to a smaller
                        value.'''.format(k, X.shape[0]))
    if batch_size > X.shape[0]:
        raise Exception('''batch_size value larger than num_rows in dataset
                        (batch_size={}, rows={}). Lower batch_size to a
                        smaller value.'''.format(batch_size, X.shape[0]))

    if Y is None:
        if precompute:
            if verbose > 0:
                print('Extracting KNN from index')

            neighbour_matrix = extract_knn(X, index_path, k=k,
                                           search_k=search_k, verbose=verbose)
            return create_knn_triplet_dataset(X, neighbour_matrix,batch_size=batch_size)
        else:
            index = AnnoyIndex(X.shape[1], metric='angular')
            index.load(index_path)
            return create_annoy_triplet_dataset(X, index, k=k,
                                                batch_size=batch_size,
                                                search_k=search_k)
    else:
        if precompute:
            if verbose > 0:
                print('Extracting KNN from index')

            neighbour_matrix = extract_knn(X, index_path, k=k,
                                           search_k=search_k, verbose=verbose)
            return create_labeled_knn_triplet_dataset(X, Y, neighbour_matrix,
                                                    batch_size=batch_size)
        else:
            index = AnnoyIndex(X.shape[1], metric='angular')
            index.load(index_path)
            return create_labeled_annoy_triplet_dataset(X, Y, index,
                                                        k=k, batch_size=batch_size,
                                                        search_k=search_k)

def create_annoy_triplet_dataset(X, annoy_index, k=150, batch_size=32, search_k=-1):
    knn_sequence = AnnoyTripletGenerator(X, annoy_index, k=k, batch_size=batch_size, search_k=search_k)

    def get_triplets_by_index(index):
        triplets, labels = knn_sequence[index]
        return (*triplets, labels)
    
    def tf_get_triplets_by_index(index):
        anchors, positives, negatives, labels = tf.py_function(
            get_triplets_by_index, [index],
            [tf.float32, tf.float32, tf.float32, tf.float32],
        )
        anchors.set_shape([None, X.shape[-1]])
        positives.set_shape([None, X.shape[-1]])
        negatives.set_shape([None, X.shape[-1]])
        labels.set_shape([None,])
        return tuple([anchors, positives, negatives]), labels
    
    index_dataset = tf.data.Dataset.from_tensor_slices([i for i in range(int(np.ceil(len(X) / batch_size)))])
    index_dataset = index_dataset.shuffle(int(np.ceil(len(X) / batch_size)))
    index_dataset = index_dataset.repeat()

    dataset = index_dataset.map(
        tf_get_triplets_by_index,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

class AnnoyTripletGenerator(Sequence):

    def __init__(self, X, annoy_index, k=150, batch_size=32, search_k=-1):
        self.X = X
        self.annoy_index = annoy_index
        self.k = k
        self.batch_size = batch_size
        self.search_k = search_k
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size,
                              min((idx + 1) * self.batch_size, self.X.shape[0]))

        placeholder_labels = self.placeholder_labels[:len(batch_indices)]
        triplet_batch = [self.knn_triplet_from_annoy_index(row_index)
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)

        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), placeholder_labels

    def knn_triplet_from_annoy_index(self, row_index):
        """ A random (unweighted) positive example chosen. """
        triplet = []
        neighbour_list = np.array(self.annoy_index.get_nns_by_item(row_index, self.k + 1,
                                  search_k=self.search_k, include_distances=False), dtype=np.uint32)

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        negative_ind = np.random.randint(0, self.X.shape[0])  # Pick a random index until one fits constraint. An optimization.
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplet += [self.X[row_index], self.X[neighbour_ind], self.X[negative_ind]]
        return triplet


def create_knn_triplet_dataset(X, neighbour_matrix, batch_size=32):
    knn_sequence = KnnTripletGenerator(X, neighbour_matrix, batch_size=batch_size)

    def get_triplets_by_index(index):
        triplets, labels = knn_sequence[index]
        return (*triplets, labels)
    
    def tf_get_triplets_by_index(index):
        anchors, positives, negatives, labels = tf.py_function(
            get_triplets_by_index, [index],
            [tf.float32, tf.float32, tf.float32, tf.float32],
        )
        anchors.set_shape([None, X.shape[-1]])
        positives.set_shape([None, X.shape[-1]])
        negatives.set_shape([None, X.shape[-1]])
        labels.set_shape([None,])
        return tuple([anchors, positives, negatives]), labels

    index_dataset = tf.data.Dataset.from_tensor_slices([i for i in range(int(np.ceil(len(X) / batch_size)))])
    index_dataset = index_dataset.shuffle(int(np.ceil(len(X) / batch_size)))
    index_dataset = index_dataset.repeat()

    dataset = index_dataset.map(
        tf_get_triplets_by_index,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class KnnTripletGenerator(Sequence):

    def __init__(self, X, neighbour_matrix, batch_size=32):
        self.X = X
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.X.shape[0]))

        placeholder_labels = self.placeholder_labels[:len(batch_indices)]
        triplet_batch = [self.knn_triplet_from_neighbour_list(row_index, self.neighbour_matrix[row_index])
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]
        triplet_batch = np.array(triplet_batch)

        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), placeholder_labels

    def knn_triplet_from_neighbour_list(self, row_index, neighbour_list):
        """ A random (unweighted) positive example chosen. """
        triplets = []

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint. An optimization.
        negative_ind = np.random.randint(0, self.X.shape[0])
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplets += [self.X[row_index], self.X[neighbour_ind],
                     self.X[negative_ind]]
        return triplets


def create_labeled_annoy_triplet_dataset(X, Y, annoy_index, k=150, batch_size=32, search_k=-1):
    knn_sequence = LabeledAnnoyTripletGenerator(X, Y, annoy_index, k=k, batch_size=batch_size, search_k=search_k)

    def get_triplets_by_index(index):
        triplets, labels = knn_sequence[index]
        return (*triplets, labels[0])
    
    def tf_get_triplets_by_index(index):
        anchors, positives, negatives, labels = tf.py_function(
            get_triplets_by_index, [index],
            [tf.float32, tf.float32, tf.float32, tf.float32],
        )
        anchors.set_shape([None, X.shape[-1]])
        positives.set_shape([None, X.shape[-1]])
        negatives.set_shape([None, X.shape[-1]])
        if Y.ndim > 1:
            labels.set_shape([None, Y.shape[-1]])
        else:
            labels.set_shape([None,])
        return tuple([anchors, positives, negatives]), tuple([labels, labels])

    index_dataset = tf.data.Dataset.from_tensor_slices([i for i in range(int(np.ceil(len(X) / batch_size)))])
    index_dataset = index_dataset.shuffle(int(np.ceil(len(X) / batch_size)))
    index_dataset = index_dataset.repeat()

    dataset = index_dataset.map(
        tf_get_triplets_by_index,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class LabeledAnnoyTripletGenerator(Sequence):

    def __init__(self, X, Y, annoy_index, k=150, batch_size=32, search_k=-1):
        self.X, self.Y = X, Y
        self.annoy_index = annoy_index
        self.k = k
        self.batch_size = batch_size
        self.search_k = search_k

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size,
                              min((idx + 1) * self.batch_size,
                                  self.X.shape[0]))

        label_batch = self.Y[batch_indices]
        triplet_batch = [self.knn_triplet_from_annoy_index(row_index)
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)

        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), tuple([np.array(label_batch), np.array(label_batch)])

    def knn_triplet_from_annoy_index(self, row_index):
        """ A random (unweighted) positive example chosen. """
        triplet = []
        neighbour_list = np.array(self.annoy_index.get_nns_by_item(row_index, self.k + 1,
                                  search_k=self.search_k, include_distances=False), dtype=np.uint32)

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint. An optimization.
        negative_ind = np.random.randint(0, self.X.shape[0])
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplet += [self.X[row_index], self.X[neighbour_ind], self.X[negative_ind]]
        return triplet


def create_labeled_knn_triplet_dataset(X, Y, neighbour_matrix, batch_size=32):
    knn_sequence = LabeledKnnTripletGenerator(X, Y, neighbour_matrix, batch_size=batch_size)

    def get_triplets_by_index(index):
        triplets, labels = knn_sequence[index]
        return (*triplets, labels[0])
    
    def tf_get_triplets_by_index(index):
        anchors, positives, negatives, labels = tf.py_function(
            get_triplets_by_index, [index],
            [tf.float32, tf.float32, tf.float32, tf.float32],
        )
        anchors.set_shape([None, X.shape[-1]])
        positives.set_shape([None, X.shape[-1]])
        negatives.set_shape([None, X.shape[-1]])
        if Y.ndim > 1:
            labels.set_shape([None, Y.shape[-1]])
        else:
            labels.set_shape([None,])
        return tuple([anchors, positives, negatives]), tuple([labels, labels])

    index_dataset = tf.data.Dataset.from_tensor_slices([i for i in range(int(np.ceil(len(X) / batch_size)))])
    index_dataset = index_dataset.shuffle(int(np.ceil(len(X) / batch_size)))
    index_dataset = index_dataset.repeat()

    dataset = index_dataset.map(
        tf_get_triplets_by_index,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class LabeledKnnTripletGenerator(Sequence):

    def __init__(self, X, Y, neighbour_matrix, batch_size=32):
        self.X, self.Y = X, Y
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.X.shape[0]))

        label_batch = self.Y[batch_indices]
        triplet_batch = [self.knn_triplet_from_neighbour_list(row_index, self.neighbour_matrix[row_index])
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)

        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), tuple([np.array(label_batch), np.array(label_batch)])

    def knn_triplet_from_neighbour_list(self, row_index, neighbour_list):
        """ A random (unweighted) positive example chosen. """
        triplets = []

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint. An optimization.
        negative_ind = np.random.randint(0, self.X.shape[0])
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplets += [self.X[row_index],
                     self.X[neighbour_ind],
                     self.X[negative_ind]]
        return triplets
