""" 
Triplet generators.

Functions for creating generators that will yield batches of triplets. 

Triplets will be created using KNNs, which can either be precomputed or dynamically generated. 
- generate_knn_triplets_from_neighbour_list will precompute KNNs
- generate_knn_triplets_from_annoy_index will dynamically generate KNNs

Where possible, precomputed KNNs are advised for speed, but where memory is a concern, dynamically generated triplets 
can be useful.

"""

from .knn import build_annoy_index, extract_knn

import numpy as np
import random

def create_triplet_generator(X, k, ntrees, batch_size, precompute=True):
    if precompute == True:
        neighbour_list = extract_knn(X, k=k, ntrees=ntrees)
        return generate_knn_triplets_from_neighbour_list(X, neighbour_list, batch_size=batch_size)
    else:
        index = build_annoy_index(X, k=k, ntrees=ntrees)
        return generate_knn_triplets_from_annoy_index(X, index, k=k, batch_size=batch_size)


def knn_triplet_from_neighbour_list(X, index, neighbour_list):
    """ A random (unweighted) positive example chosen. """
    triplets = []
    
    # Take a random neighbour as positive
    neighbour = neighbour_list[np.random.choice(range(len(neighbour_list)))]
    
    # Take a random non-neighbour as negative
    negative_ind = random.randrange(0, len(X))
    while negative_ind == index or negative_ind in neighbour_list:
        negative_ind = random.randrange(0, len(X))
    triplets += [[X[index], X[neighbour], X[negative_ind]]]
    return triplets


def generate_knn_triplets_from_neighbour_list(X, neighbour_list, batch_size=32):
    iterations = 0
    row_indexes = list(range(len(X)))
    np.random.shuffle(row_indexes)
    
    while True:
        triplet_batch = []
        placeholder_labels = np.array([0 for i in range(batch_size)])
        
        for i in range(batch_size):
            if iterations >= len(X):
                np.random.shuffle(row_indexes)
                iterations = 0
           
            triplet = knn_triplet_from_neighbour_list(X, row_indexes[iterations], neighbour_list[row_indexes[iterations]])
            triplet_batch += triplet
            iterations += 1
        
        triplet_batch = np.array(triplet_batch)
        yield ([triplet_batch[:,0], triplet_batch[:,1], triplet_batch[:,2]], placeholder_labels)

def knn_triplet_from_annoy_index(X, annoy_index, row_index, k):
    """ A random (unweighted) positive example chosen. """
    triplets = []
    neighbour_list = np.array(annoy_index.get_nns_by_item(row_index, k+1, search_k=-1, include_distances=False), dtype=np.uint32)
    
    # Take a random neighbour as positive
    neighbour = neighbour_list[np.random.choice(range(len(neighbour_list)))]
    
    # Take a random non-neighbour as negative
    negative_ind = random.randrange(0, len(X))
    while negative_ind == row_index or negative_ind in neighbour_list:
        negative_ind = random.randrange(0, len(X))
    triplets += [[X[row_index], X[neighbour], X[negative_ind]]]
    return triplets

def generate_knn_triplets_from_annoy_index(X, annoy_index, k=150, batch_size=32):
    iterations = 0
    row_indexes = list(range(len(X)))
    np.random.shuffle(row_indexes)
    
    while True:
        triplet_batch = []
        placeholder_labels = np.array([0 for i in range(batch_size)])
        
        for i in range(batch_size):
            if iterations >= len(X):
                np.random.shuffle(row_indexes)
                iterations = 0                    
            
            triplet = knn_triplet_from_annoy_index(X, annoy_index, row_indexes[iterations], k=k)
            triplet_batch += triplet
            iterations += 1
        
        triplet_batch = np.array(triplet_batch)
        yield ([triplet_batch[:,0], triplet_batch[:,1], triplet_batch[:,2]], placeholder_labels)

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

