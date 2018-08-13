import louvain
import numpy as np

from annoy import AnnoyIndex


def extract_knn(X, ntrees=50, k=150):
    f = X.shape[1]
    t = AnnoyIndex(f)
    for i in range(X.shape[0]):
        v = X[i,:] 
        t.add_item(i, v)

    # Build n trees
    t.build(ntrees)

    def knn(x, k = k):
        k = t.get_nns_by_item(x, k+1, include_distances=False) 
        return k

    edge_list = []
    for element in range(X.shape[0]):
        edge_list.append(knn(element, k=k))
    
    return np.array(edge_list, dtype=np.uint32)
