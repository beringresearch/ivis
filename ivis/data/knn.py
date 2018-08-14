""" KNN retrieval using an Annoy index. """

import numpy as np

from annoy import AnnoyIndex

def build_annoy_index(X, k=150, ntrees=50):
    index = AnnoyIndex(X.shape[1])
    for i in range(X.shape[0]):
        v = X[i,:] 
        index.add_item(i, v)

    # Build n trees
    index.build(ntrees)
    return index

def extract_knn(X, k=150, ntrees=50, search_k=-1):
    index = build_annoy_index(X, k=k, ntrees=ntrees)

    def knn(x, k = k):
        k = index.get_nns_by_item(x, k+1, search_k=search_k, include_distances=False) 
        return np.array(k, dtype=np.uint32)

    edge_list = []
    for element in range(X.shape[0]):
        edge_list.append(knn(element, k=k))
    
    return edge_list
