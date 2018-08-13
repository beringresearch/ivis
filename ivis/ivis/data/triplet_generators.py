"""

"""

import numpy as np
import random

def create_triplets_from_positive_index_dict(x_train, positive_index_dict):
    triplets = []
    labels_placeholder = []
    for i in range(len(x_train)):
        try:
            for neighbour in positive_index_dict[i]:
                ind = i
                while ind == i or ind in positive_index_dict[i]:
                    ind = random.randrange(0, len(x_train))
                triplets += [[x_train[i], x_train[neighbour], x_train[ind]]]
                labels_placeholder += [1]
        except:
            pass
    return np.array(triplets), np.array(labels_placeholder)


def create_knn_triplet(x_train, index, neighbour_list):
    """ A random (unweighted) positive example chosen. """
    triplets = []
    
    # Take a random neighbour as positive
    neighbour = neighbour_list[np.random.choice(range(len(neighbour_list)))]
    
    # Take a random non-neighbour as negative
    negative_ind = random.randrange(0, len(x_train))
    while negative_ind == index or negative_ind in neighbour_list:
        negative_ind = random.randrange(0, len(x_train))
    triplets += [[x_train[index], x_train[neighbour], x_train[negative_ind]]]
    return triplets


def generate_knn_triplets(X, neighbour_list, batch_size=32):
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
           
            triplet = create_knn_triplet(X, row_indexes[iterations], neighbour_list[row_indexes[iterations]])
            triplet_batch += triplet
            iterations += 1
        
        triplet_batch = np.array(triplet_batch)
        yield ([triplet_batch[:,0], triplet_batch[:,1], triplet_batch[:,2]], placeholder_labels)

