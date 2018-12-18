""" KNN retrieval using an Annoy index. """

import numpy as np
from scipy.sparse import issparse
from annoy import AnnoyIndex
from tqdm import trange
from multiprocessing import Process, RawArray, Pool, cpu_count, Queue
from functools import reduce, partial

def build_annoy_index(X, ntrees=50):
    print('Building KNN index')
    
    if issparse(X): X = X.toarray()
    index = AnnoyIndex(X.shape[1])
    for i in range(X.shape[0]):
        v = X[i] 
        index.add_item(i, v)

    # Build n trees
    index.build(ntrees)
    return index

def extract_knn(X, index_filepath, k=150, search_k=-1):
    """ Starts multiple processes to retrieve nearest neighbours using an Annoy Index in parallel """
    
    print('Extracting KNN from index')

    n_dims = X.shape[1]

    chunk_size = len(X) // cpu_count()
    remainder = (len(X) % cpu_count()) > 0
    process_pool = []
    results_queue_list = []

    i = 0
    while (i + chunk_size) <= len(X):
        results_queue = Queue()
        process_pool.append(KNN_Worker(index_filepath, k, search_k, n_dims, (i, i+chunk_size), results_queue))
        results_queue_list.append(results_queue)
        i += chunk_size
    if remainder:
        results_queue = Queue()
        process_pool.append(KNN_Worker(index_filepath, k, search_k, n_dims, (i, len(X)), results_queue))
        results_queue_list.append(results_queue)

    for process in process_pool:
        process.start()
    
    for process in process_pool:
        process.join()

    neighbour_list = []
    for queue in results_queue_list:
        while not queue.empty():
            neighbour_list.append(queue.get())

    return np.array(neighbour_list)

class KNN_Worker(Process):
    """
    Upon construction, this worker process loads an annoy index from disk. Upon being started, the neighbours of the 
    data-points specified by 'data_indices' variable will be retrieved from the index according to 
    the provided parameters and stored in the 'results_queue'.

    data_indices is a tuple of integers denoting the start and end range of indices to retrieve
    """
    def __init__(self, index_filepath, k, search_k, n_dims, data_indices, results_queue):
        self.index = AnnoyIndex(n_dims)
        self.index.load(index_filepath)
        self.k = k
        self.search_k = search_k
        self.data_indices = data_indices
        self.results_queue = results_queue
        super(KNN_Worker, self).__init__()

    def run(self):
        for i in range(self.data_indices[0], self.data_indices[1]):
            try:
                print('{} adding to queue'.format(i))
                neighbour_indexes = self.index.get_nns_by_item(i, self.k+1, search_k=self.search_k, include_distances=False)
                print(neighbour_indexes)
                neighbour_indexes = np.array(neighbour_indexes, dtype=np.uint32)
                self.results_queue.put(neighbour_indexes)
            except Exception as e:
                print(e)
        self.results_queue.close()