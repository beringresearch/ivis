""" KNN retrieval using an Annoy index. """

import time
from multiprocessing import Process, cpu_count, Queue
from collections import namedtuple
from operator import attrgetter
from abc import ABC, abstractmethod
from scipy.sparse import issparse
from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np


def build_annoy_index(X, path, metric='angular', ntrees=50, build_index_on_disk=True, verbose=1):
    """ Build a standalone annoy index.

    :param array X: numpy array with shape (n_samples, n_features)
    :param str path: The filepath of a trained annoy index file
        saved on disk.
    :param int ntrees: The number of random projections trees built by Annoy to
        approximate KNN. The more trees the higher the memory usage, but the
        better the accuracy of results.
    :param bool build_index_on_disk: Whether to build the annoy index directly
        on disk. Building on disk should allow for bigger datasets to be indexed,
        but may cause issues.
    :param str metric: Which distance metric Annoy should use when building KNN index.
        Supports "angular", "euclidean", "manhattan", "hamming", or "dot".
    :param int verbose: Controls the volume of logging output the model
        produces when training. When set to 0, silences outputs, when above 0
        will print outputs.

    """

    if verbose:
        print("Building KNN index")

    if "reshape" in dir(X):
        X = X.reshape((X.shape[0], -1))

    index = AnnoyIndex(X.shape[1], metric=metric)
    if build_index_on_disk:
        index.on_disk_build(path)

    if issparse(X):
        for i in tqdm(range(X.shape[0]), disable=verbose < 1):
            vector = X[i].toarray()[0]
            index.add_item(i, vector)
    else:
        for i in tqdm(range(X.shape[0]), disable=verbose < 1):
            vector = X[i]
            index.add_item(i, vector)

    try:
        index.build(ntrees)
    except Exception as e:
        msg = ("Error building Annoy Index. Passing on_disk_build=False"
               " may solve the issue, especially on Windows.")
        raise Exception(msg) from e
    else:
        if not build_index_on_disk:
            index.save(path)
        return index


class NeighbourMatrix(ABC):
    r"""A matrix |A_ij| where i is the row index of the data point and j
    refers to the index of the neigbouring point.

    .. |A_ij| replace:: A\ :subscript:`ij`"""
    @abstractmethod
    def __len__(self):
        raise NotImplementedError
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
    def get_neighbour_indices(self):
        """Retrieves all neighbours for all points"""
        return [self.__getitem__(i) for i in self.__len__()]


class AnnoyKnnMatrix(NeighbourMatrix):
    r"""A matrix |A_ij| where i is the row index of the data point and j
    refers to the index of the neigbouring point. Neighbouring points
    are KNN retrieved using an Annoy Index.

    .. |A_ij| replace:: A\ :subscript:`ij`"""

    def __init__(self, index, shape, index_path='annoy.index', k=150, search_k=-1,
                 precompute=False, include_distances=False, verbose=False):
        if k >= shape[0] - 1:
            raise ValueError('''k value greater than or equal to (num_rows - 1)
                             (k={}, rows={}). Lower k to a smaller
                             value.'''.format(k, shape[0]))
        self.index = index
        self.shape = shape
        self.index_path = index_path
        self.k = k
        self.search_k = search_k
        self.include_distances = include_distances
        self.verbose = verbose
        self.precomputed_neighbours = None
        if precompute:
            self.precomputed_neighbours = self.get_neighbour_indices()

    @classmethod
    def build(cls, X, path, k=150, metric='angular', search_k=-1, include_distances=False,
              ntrees=50, build_index_on_disk=True, precompute=False, verbose=1):
        """Builds a new Annoy Index on input data *X*, then constructs and returns an
        AnnoyKnnMatrix object using the newly-built index."""

        index = build_annoy_index(X, path, metric, ntrees, build_index_on_disk, verbose)
        return cls(index, X.shape, path, k, search_k, precompute, include_distances, verbose)

    @classmethod
    def load(cls, index_path, shape, k=150, search_k=-1, include_distances=False,
             precompute=False, verbose=1):
        """Constructs and returns an AnnoyKnnMatrix object from an existing Annoy Index on disk."""

        index = AnnoyIndex(shape[1], metric='angular')
        index.load(index_path)
        return cls(index, shape, index_path, k, search_k, precompute, include_distances, verbose)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if self.precomputed_neighbours is not None:
            return self.precomputed_neighbours[idx]
        return self.index.get_nns_by_item(
            idx, self.k + 1, search_k=self.search_k, include_distances=self.include_distances)

    def get_neighbour_indices(self):
        if self.precomputed_neighbours is not None:
            return self.precomputed_neighbours
        return extract_knn(
            self.shape, k=self.k,
            index_builder=self.load,
            verbose=self.verbose,
            index_path=self.index_path,
            search_k=self.search_k,
            include_distances=self.include_distances)


IndexNeighbours = namedtuple('IndexNeighbours', ['row_index', 'neighbour_list'])
def extract_knn(data_shape, index_builder=AnnoyKnnMatrix.load, verbose=1, **kwargs):
    """Starts multiple processes to retrieve nearest neighbours from a built index in parallel.

    :param tuple data_shape: The shape of the data that the index was built on.
    :param index_builder Callable: Called within each worker process to load the index.
    :param verbose int: Controls verbosity of output.
    :param kwargs: keyword arguments to pass to the index_builder function."""

    if verbose:
        print("Extracting KNN neighbours")

    chunk_size = data_shape[0] // cpu_count()
    process_pool = []
    results_queue = Queue()
    error_queue = Queue()

    # Split up the indices and assign processes for each chunk
    for i in range(0, data_shape[0], chunk_size):
        process_pool.append(KnnWorker(results_queue, (i, min(i+chunk_size, data_shape[0])),
                                      index_builder, shape=data_shape,
                                      error_queue=error_queue, **kwargs))
    try:
        for process in process_pool:
            process.start()

        neighbour_list = []
        neighbour_list_length = len(neighbour_list)

        with tqdm(total=data_shape[0], disable=verbose < 1) as pbar:
            # Read from queue constantly to prevent it from becoming full
            while any(process.is_alive() for process in process_pool):
                # Check for errors in worker processes
                if not error_queue.empty():
                    raise error_queue.get()
                while not results_queue.empty():
                    neighbour_list.append(results_queue.get())
                progress = len(neighbour_list) - neighbour_list_length
                pbar.update(progress)
                neighbour_list_length = len(neighbour_list)
                time.sleep(0.1)

            # Read any remaining results
            while not results_queue.empty():
                neighbour_list.append(results_queue.get())

        # Final check for errors in worker processes
        if not error_queue.empty():
            raise error_queue.get()

        neighbour_list = sorted(neighbour_list, key=attrgetter('row_index'))
        neighbour_list = list(map(attrgetter('neighbour_list'), neighbour_list))

        return np.array(neighbour_list)

    except:
        print('Halting KNN retrieval and cleaning up')
        for process in process_pool:
            process.terminate()
        raise


class KnnWorker(Process):
    """When run, this worker process loads an index from disk using the provided
    'knn_init' function, passing all additional 'kwargs' to this initializer function.
    When started, the neighbours of the data-points between the range specified by `data_indices`
    and stored in the 'results_queue'.

    Each result will be send to the result queue as a named tuple, 'IndexNeighbours'
    with field names ['row_index', 'neighbour_list'].

    :param: multiprocessing.Queue results_queue. Queue worker will push results to.
    :param: tuple data_indices. Specifies range (start, end) to retrieve neighbours for.
    :param: Callable knn_init. Function to load index from disk.
    :param: Optional[multiprocessing.Queue]. If exception encountered, will be pushed to this
        queue if provided.
    :param: **kwargs. Args to pass to knn_init function.
    """

    def __init__(self, results_queue, data_indices, knn_init, error_queue=None, **kwargs):
        self.results_queue = results_queue
        self.data_indices = data_indices
        self.knn_init = knn_init
        self.error_queue = error_queue
        self.kwargs = kwargs
        super().__init__()

    def run(self):
        """Load index, retrieve neighbouring points, and send to results queue."""
        try:
            annoy_neighbours = self.knn_init(**self.kwargs)
            for i in range(self.data_indices[0], self.data_indices[1]):
                neighbour_indexes = annoy_neighbours[i]
                self.results_queue.put(
                    IndexNeighbours(row_index=i,
                                    neighbour_list=neighbour_indexes))
        except Exception as e:
            if self.error_queue:
                self.error_queue.put(e)
                self.error_queue.close()
        finally:
            self.results_queue.close()
