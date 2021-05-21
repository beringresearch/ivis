""" KNN retrieval using an Annoy index. """

import functools
import time
from multiprocessing import Array, Process, Value, cpu_count, Queue
from collections.abc import Sequence
from operator import attrgetter
from scipy.sparse import issparse
from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np


class AnnoyKnnMatrix(Sequence):
    r"""A matrix |A_ij| where i is the row index of the data point and j
    refers to the index of the neigbouring point. Neighbouring points
    are KNN retrieved using an Annoy Index.

    .. |A_ij| replace:: A\ :subscript:`ij`

    :param AnnoyIndex index: AnnoyIndex instance to use when retrieving KNN
    :param tuple shape: Shape of the KNN matrix (height, width)
    :param string index_path: Location of the AnnoyIndex file on disk
    :param int k: The number of neighbours to retrieve for each point
    :param int search_k: Controls the number of nodes searched - higher is more
        expensive but more accurate. Default of -1 defaults to n_trees * k
    :param boolean precompute: Whether to precompute the KNN index and store the matrix in memory.
        Much faster when training, but consumes more memory.
    :param boolean include_distances: Whether to return the distances along with the indexes of
        the neighbouring points
    :param boolean verbose: Controls verbosity of output to console when building index. If
        False, nothing will be printed to the terminal."""

    def __init__(self, index, shape, index_path='annoy.index', k=150, search_k=-1,
                 precompute=False, include_distances=False, verbose=False):
        """Constructs an AnnoyKnnMatrix instance from an AnnoyIndex object with given parameters"""
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

    def __del__(self):
        self.index.unload()

    @classmethod
    def build(cls, X, path, k=150, metric='angular', search_k=-1, include_distances=False,
              ntrees=50, build_index_on_disk=True, precompute=False, verbose=1):
        """Builds a new Annoy Index on input data *X*, then constructs and returns an
        AnnoyKnnMatrix object using the newly-built index."""

        _validate_knn_shape(X.shape, k)
        index = build_annoy_index(X, path, metric, ntrees, build_index_on_disk, verbose)
        return cls(index, X.shape, path, k, search_k, precompute, include_distances, verbose)

    @classmethod
    def load(cls, index_path, shape, k=150, search_k=-1, include_distances=False,
             precompute=False, verbose=1):
        """Constructs and returns an AnnoyKnnMatrix object from an existing Annoy Index on disk."""

        _validate_knn_shape(shape, k)
        index = AnnoyIndex(shape[1], metric='angular')
        index.load(index_path)
        return cls(index, shape, index_path, k, search_k, precompute, include_distances, verbose)

    def __len__(self):
        """Number of rows in neighbour matrix"""
        return self.shape[0]

    def __getitem__(self, idx):
        """Returns neighbours list for the specified index.
        Supports both integer and slice indices."""
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        if idx < 0:
            idx += len(self)
        if self.precomputed_neighbours is not None:
            return self.precomputed_neighbours[idx]
        return self.index.get_nns_by_item(
            idx, self.k, search_k=self.search_k, include_distances=self.include_distances)

    def get_neighbour_indices(self):
        """Retrieves neighbours for every row in parallel"""
        if self.precomputed_neighbours is not None:
            return self.precomputed_neighbours
        return extract_knn(
            self.shape, k=self.k,
            index_builder=self.load,
            verbose=self.verbose,
            index_path=self.index_path,
            search_k=self.search_k,
            include_distances=self.include_distances)


def _validate_knn_shape(shape, k):
    if k >= shape[0]:
        raise ValueError('''k value greater than or equal to num_rows
                            (k={}, rows={}). Lower k to a smaller
                            value.'''.format(k, shape[0]))

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

    if len(X.shape) > 2:
        if not "reshape" in dir(X):
            raise ValueError("Attempting to build AnnoyIndex on multi-dimensional data"
                             " without providing a reshape method. AnnoyIndexes require"
                             " 2D data - rows and columns.")
        if verbose:
            print('Flattening multidimensional input before building KNN index using Annoy')
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

    if not build_index_on_disk:
        index.save(path)
    return index

def extract_knn(data_shape, k, index_builder=AnnoyKnnMatrix.load, verbose=1, **kwargs):
    """Starts multiple processes to retrieve nearest neighbours from a built index in parallel.

    :param int k: Number of neighbours to retrieve. Passed as a kwarg to index builder function.
    :param tuple data_shape: The shape of the data that the index was built on.
    :param Callable index_builder: Called within each worker process to load the index.
    :param int verbose: Controls verbosity of output.
    :param kwargs: keyword arguments to pass to the index_builder function."""

    if verbose:
        print("Extracting KNN neighbours")

    if len(data_shape) > 2:
        if verbose:
            print('Flattening data before retrieving KNN from index')
        data_shape = (data_shape[0], functools.reduce(lambda x, y: x*y, data_shape[1:]))

    array_ctype = _get_uint_ctype(data_shape[0])
    neighbours_array = Array(array_ctype, data_shape[0] * k, lock=False)

    chunk_size = max(1, data_shape[0] // cpu_count())
    process_pool = []
    progress_counters = []
    error_queue = Queue()

    # Split up the indices and assign processes for each chunk
    for i in range(0, data_shape[0], chunk_size):
        counter = Value(array_ctype, 0, lock=False)
        process_pool.append(KnnWorker(neighbours_array, (i, min(i+chunk_size, data_shape[0])),
                                      index_builder, k=k, shape=data_shape, error_queue=error_queue,
                                      progress_counter=counter, **kwargs))
        progress_counters.append(counter)
    try:
        for process in process_pool:
            process.start()

        # Poll for progress updates
        with tqdm(total=data_shape[0], disable=verbose < 1) as pbar:
            while pbar.n < data_shape[0]:
                # Raise worker errors
                if not error_queue.empty():
                    raise error_queue.get()

                num_processed = sum([num.value for num in progress_counters])
                pbar.update(num_processed - pbar.n)
                time.sleep(0.1)

        # Join processes to avoid zombie processes on UNIX
        for process in process_pool:
            process.join()

        neighbour_matrix = np.ndarray((data_shape[0], k), buffer=neighbours_array,
                                      dtype=array_ctype)
        return neighbour_matrix

    except:
        print('Halting KNN retrieval and cleaning up')
        for process in process_pool:
            process.terminate()
            process.join()
        raise


class KnnWorker(Process):
    """When run, this worker process loads an index from disk using the provided
    'knn_init' function, passing all additional 'kwargs' to this initializer function.
    Then the neighbours of the data-points between the range specified by `data_indices`
    are retrieved and stored in the 'neighbours_array'.

    :param: multiprocessing.Array neighbours_array. 1D array that worker will insert results into.
    :param: tuple data_indices. Specifies range (start, end) to retrieve neighbours for.
    :param: Callable knn_init. Function to load index from disk.
    :param: int k. Number of neighbours to retrieve. Will be passed as kwargs to knn_init function.
    :param: Optional[multiprocessing.Value] progress_counter. Incremented for every row processed.
    :param: Optional[multiprocessing.Queue] error_queue. Exceptions encountered are sent to this
        queue if provided.
    :param: **kwargs. Args to pass to knn_init function.
    """

    def __init__(self, neighbours_array, data_indices, knn_init, k,
                 progress_counter=None, error_queue=None, **kwargs):
        super().__init__()
        self.neighbours_array = neighbours_array
        self.data_indices = data_indices
        self.k = k
        self.knn_init = knn_init
        self.progress_counter = progress_counter
        self.error_queue = error_queue
        self.kwargs = kwargs
        self.kwargs['k'] = k

    def run(self):
        """Load index, retrieve neighbouring points, and insert results into 1D array.
        Can update progress_counter and send Exceptions to error queue if provided."""
        try:
            knn_index = self.knn_init(**self.kwargs)

            for i in range(self.data_indices[0], self.data_indices[1]):
                neighbour_indexes = knn_index[i]
                row_offset = i * self.k
                for j in range(0, self.k):
                    self.neighbours_array[row_offset + j] = neighbour_indexes[j]
                if self.progress_counter is not None:
                    self.progress_counter.value += 1
        except Exception as e:
            if self.error_queue:
                self.error_queue.put(e)
            raise

def _get_uint_ctype(integer):
    """Gets smallest possible uint representation of provided integer.
    Raises ValueError if invalid value provided (negative or above uint64)."""
    for dtype in np.typecodes["UnsignedInteger"]:
        min_val, max_val = attrgetter('min', 'max')(np.iinfo(np.dtype(dtype)))
        if min_val <= integer <= max_val:
            return dtype
    raise ValueError("Cannot parse {} as a uint64".format(integer))
