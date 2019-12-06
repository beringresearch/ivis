""" KNN retrieval using an Annoy index. """

import numpy as np
from scipy.sparse import issparse
from annoy import AnnoyIndex
from multiprocessing import Process, cpu_count, Queue
from collections import namedtuple
from operator import attrgetter
from tqdm import tqdm
import time


def build_annoy_index(X, path, ntrees=50, build_index_on_disk=True, verbose=1):
    """ Build a standalone annoy index.

    :param array X: numpy array with shape (n_samples, n_features)
    :param str path: The filepath of a trained annoy index file
        saved on disk.
    :param int ntrees: The number of random projections trees built by Annoy to
        approximate KNN. The more trees the higher the memory usage, but the
        better the accuracy of results.
    :param bool build_index_on_disk: Whether to build the annoy index directly
        on disk. Building on disk should allow for bigger datasets to be indexed,
        but may cause issues. If None, on-disk building will be enabled for Linux, 
        but not Windows due to issues on Windows.
    :param int verbose: Controls the volume of logging output the model
        produces when training. When set to 0, silences outputs, when above 0
        will print outputs.

    """

    index = AnnoyIndex(X.shape[1], metric='angular')
    if build_index_on_disk:
        index.on_disk_build(path)

    if issparse(X):
        for i in tqdm(range(X.shape[0]), disable=verbose < 1):
            v = X[i].toarray()[0]
            index.add_item(i, v)
    else:
        for i in tqdm(range(X.shape[0]), disable=verbose < 1):
            v = X[i]
            index.add_item(i, v)
    
    try:
        index.build(ntrees)
    except Exception:
        msg = ("Error building Annoy Index. Passing on_disk_build=False"
                " may solve the issue, especially on Windows.")
        raise IndexBuildingError(msg)
    else:
        if not build_index_on_disk:
            index.save(path)
        return index


def extract_knn(X, index_filepath, k=150, search_k=-1, verbose=1):
    """ Starts multiple processes to retrieve nearest neighbours using
        an Annoy Index in parallel """

    n_dims = X.shape[1]

    chunk_size = X.shape[0] // cpu_count()
    remainder = (X.shape[0] % cpu_count()) > 0
    process_pool = []
    results_queue = Queue()

    # Split up the indices and assign processes for each chunk
    i = 0
    while (i + chunk_size) <= X.shape[0]:
        process_pool.append(KNN_Worker(index_filepath, k, search_k, n_dims,
                                       (i, i+chunk_size), results_queue))
        i += chunk_size
    if remainder:
        process_pool.append(KNN_Worker(index_filepath, k, search_k, n_dims,
                                       (i, X.shape[0]), results_queue))

    try:
        for process in process_pool:
            process.start()

        # Read from queue constantly to prevent it from becoming full
        with tqdm(total=X.shape[0], disable=verbose < 1) as pbar:
            neighbour_list = []
            neighbour_list_length = len(neighbour_list)
            while any(process.is_alive() for process in process_pool):
                while not results_queue.empty():
                    neighbour_list.append(results_queue.get())
                progress = len(neighbour_list) - neighbour_list_length
                pbar.update(progress)
                neighbour_list_length = len(neighbour_list)
                time.sleep(0.1)

            while not results_queue.empty():
                neighbour_list.append(results_queue.get())

        neighbour_list = sorted(neighbour_list, key=attrgetter('row_index'))
        neighbour_list = list(map(attrgetter('neighbour_list'), neighbour_list))

        return np.array(neighbour_list)
    
    except:
        print('Halting KNN retrieval and cleaning up')
        for process in process_pool:
            process.terminate()
        raise


IndexNeighbours = namedtuple('IndexNeighbours', 'row_index neighbour_list')


class KNN_Worker(Process):
    """
    Upon construction, this worker process loads an annoy index from disk.
    When started, the neighbours of the data-points specified by `data_indices`
    will be retrieved from the index according to the provided parameters
    and stored in the 'results_queue'.

    `data_indices` is a tuple of integers denoting the start and end range of
    indices to retrieve.
    """
    def __init__(self, index_filepath, k, search_k, n_dims,
                 data_indices, results_queue):
        self.index_filepath = index_filepath
        self.k = k
        self.n_dims = n_dims
        self.search_k = search_k
        self.data_indices = data_indices
        self.results_queue = results_queue
        super(KNN_Worker, self).__init__()

    def run(self):
        try:
            index = AnnoyIndex(self.n_dims, metric='angular')
            index.load(self.index_filepath)
            for i in range(self.data_indices[0], self.data_indices[1]):
                neighbour_indexes = index.get_nns_by_item(
                    i, self.k, search_k=self.search_k, include_distances=False)
                neighbour_indexes = np.array(neighbour_indexes,
                                                dtype=np.uint32)
                self.results_queue.put(
                    IndexNeighbours(row_index=i,
                                    neighbour_list=neighbour_indexes))
        except Exception as e:
            self.exception = e
        finally:
            self.results_queue.close()


class IndexBuildingError(OSError):
    pass