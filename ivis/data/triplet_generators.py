""" 
Triplet generators.

Functions for creating generators that will yield batches of triplets. 

Triplets will be created using KNNs, which can either be precomputed or dynamically generated. 
- generate_knn_triplets_from_neighbour_list will precompute KNNs
- generate_knn_triplets_from_annoy_index will dynamically generate KNNs

Where possible, precomputed KNNs are advised for speed, but where memory is a concern, dynamically generated triplets 
can be useful.

"""

from .knn import extract_knn
from scipy.sparse import issparse

import numpy as np
import threading


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return 

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)

    def next(self): # Py2
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def create_triplet_generator_from_annoy_index(X, index, k, batch_size, search_k=-1, precompute=True):
    if k >= len(X) - 1:
        raise Exception('k value greater than or equal to (num_rows - 1) (k={}, rows={}). Lower k to a smaller value.'.format(k, len(X)))
    if batch_size > len(X):
        raise Exception('batch_size value larger than num_rows in dataset (batch_size={}, rows={}). Lower batch_size to a smaller value.'.format(batch_size, len(X)))
    
    if precompute == True:
        neighbour_list = extract_knn(X, index, k=k, search_k=search_k)
        return generate_knn_triplets_from_neighbour_list(X, neighbour_list, batch_size=batch_size)
    else:
        return generate_knn_triplets_from_annoy_index(X, index, k=k, batch_size=batch_size, search_k=search_k)


def create_triplet_generator_from_labels(X, y, batch_size):
    return generate_triplets_from_labels(X, np.array(y), batch_size=batch_size)


def knn_triplet_from_neighbour_list(X, index, neighbour_list):
    """ A random (unweighted) positive example chosen. """
    triplets = []
    
    # Take a random neighbour as positive
    neighbour_ind = np.random.choice(neighbour_list)
    
    # Take a random non-neighbour as negative
    negative_ind = np.random.randint(0, len(X))     # Pick a random index until one fits constraint. An optimization.
    while negative_ind in neighbour_list:
        negative_ind = np.random.randint(0, len(X))
    
    triplets += [[X[index], X[neighbour_ind], X[negative_ind]]]
    return triplets

@threadsafe_generator
def generate_knn_triplets_from_neighbour_list(X, neighbour_list, batch_size=32):
    iterations = 0
    row_indexes = np.array(list(range(len(X))), dtype=np.uint32)
    np.random.shuffle(row_indexes)

    placeholder_labels = np.array([0 for i in range(batch_size)])

    while True:
        triplet_batch = []
        
        for i in range(batch_size):
            if iterations >= len(X):
                np.random.shuffle(row_indexes)
                iterations = 0
           
            triplet = knn_triplet_from_neighbour_list(X, row_indexes[iterations], neighbour_list[row_indexes[iterations]])
            triplet_batch += triplet
            iterations += 1
        
        if (issparse(triplet_batch)): triplet_batch = triplet_batch.toarray()
        triplet_batch = np.array(triplet_batch)        
        yield ([triplet_batch[:,0], triplet_batch[:,1], triplet_batch[:,2]], placeholder_labels)

def knn_triplet_from_annoy_index(X, annoy_index, row_index, k, search_k=-1):
    """ A random (unweighted) positive example chosen. """
    triplets = []
    neighbour_list = np.array(annoy_index.get_nns_by_item(row_index, k+1, search_k=-1, include_distances=False), dtype=np.uint32)
    
    # Take a random neighbour as positive
    neighbour_ind = np.random.choice(neighbour_list)
    
    # Take a random non-neighbour as negative
    negative_ind = np.random.randint(0, len(X))     # Pick a random index until one fits constraint. An optimization.
    while negative_ind in neighbour_list:
        negative_ind = np.random.randint(0, len(X))

    triplets += [[X[row_index], X[neighbour_ind], X[negative_ind]]]
    return triplets

@threadsafe_generator
def generate_knn_triplets_from_annoy_index(X, annoy_index, k=150, batch_size=32, search_k=-1):
    iterations = 0
    row_indexes = np.array(list(range(len(X))), dtype=np.uint32)
    np.random.shuffle(row_indexes)

    placeholder_labels = np.array([0 for i in range(batch_size)])
    
    while True:
        triplet_batch = []
        
        for i in range(batch_size):
            if iterations >= len(X):
                np.random.shuffle(row_indexes)
                iterations = 0                    
            
            triplet = knn_triplet_from_annoy_index(X, annoy_index, row_indexes[iterations], k=k, search_k=search_k)
            triplet_batch += triplet
            iterations += 1
        
        if (issparse(triplet_batch)): triplet_batch = triplet_batch.toarray()
        triplet_batch = np.array(triplet_batch)
        yield ([triplet_batch[:,0], triplet_batch[:,1], triplet_batch[:,2]], placeholder_labels)

@threadsafe_generator
def generate_triplets_from_labels(X, Y, batch_size=32):
    iterations = 0
    row_indexes = np.array(list(range(len(X))), dtype=np.uint32)
    np.random.shuffle(row_indexes)

    placeholder_labels = np.array([0 for i in range(batch_size)])

    while True:
        triplet_batch = []
        
        for i in range(batch_size):
            if iterations >= len(X):
                np.random.shuffle(row_indexes)
                iterations = 0
           
            triplet = triplet_from_labels(X, Y, row_indexes[iterations])
            triplet_batch += triplet
            iterations += 1
        
        if (issparse(triplet_batch)): triplet_batch = triplet_batch.toarray()
        triplet_batch = np.array(triplet_batch)
        yield ([triplet_batch[:,0], triplet_batch[:,1], triplet_batch[:,2]], placeholder_labels)

def triplet_from_labels(X, Y, index):
    """ A random (unweighted) positive example chosen. """
    triplets = []

    row_label = Y[index]
    neighbour_indexes = np.where(Y == row_label)[0]
    
    # Take a random neighbour as positive
    neighbour_ind = np.random.choice(neighbour_indexes)
    
    # Take a random non-neighbour as negative
    negative_ind = np.random.randint(0, len(X))     # Pick a random index until one fits constraint. An optimization.
    while negative_ind in neighbour_indexes:
        negative_ind = np.random.randint(0, len(X))
    
    triplets += [[X[index], X[neighbour_ind], X[negative_ind]]]
    return triplets

def create_triplets_from_positive_index_dict(X, positive_index_dict):
    triplets = []
    labels_placeholder = []
    for i in range(len(X)):
        try:
            for neighbour in positive_index_dict[i]:
                ind = i
                while ind == i or ind in positive_index_dict[i]:
                    ind = random.randrange(0, len(X))
                triplets += [[X[i], X[neighbour], X[ind]]]
                labels_placeholder += [1]
        except:
            pass
    return np.array(triplets), np.array(labels_placeholder)

