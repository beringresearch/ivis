""" KNN retrieval using an Annoy index. """

from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import time
from abc import abstractmethod
from threading import Thread
from collections.abc import Sequence
from pathlib import Path
from scipy.sparse import issparse
from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np

from ..data import get_uint_ctype

class NeighbourMatrix(Sequence):
    r"""A matrix A\ :subscript:`ij` where i is the row index of the data point and j
    refers to the index of the neigbouring point.

    """
    @property
    @abstractmethod
    def k(self):
        """The width of the matrix (number of neighbours retrieved)"""
        raise NotImplementedError

    def get_batch(self, idx_seq):
        """Gets a batch of neighbours corresponding to the provided index sequence.

        Non-optimized version, can be overridden by child classes to be made be efficient"""
        return [self.__getitem__(item) for item in idx_seq]

class AnnoyKnnMatrix(NeighbourMatrix):
    r"""Neighbouring points are KNN retrieved using an Annoy Index.

    :param AnnoyIndex index: AnnoyIndex instance to use when retrieving KNN
    :param tuple nrows: Number of rows in data matrix was built on
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

    __doc__ = NeighbourMatrix.__doc__ + __doc__
    k = None

    def __init__(self, index, nrows, index_path='annoy.index', metric='angular', k=150, search_k=-1,
                 precompute=False, include_distances=False, verbose=False, n_jobs=-1):
        """Constructs an AnnoyKnnMatrix instance from an AnnoyIndex object with given parameters"""
        self.index = index
        self.nrows = nrows
        self.index_dims = index.f
        self.metric = metric
        self.index_path = index_path
        self.k = k
        self.search_k = search_k
        self.include_distances = include_distances
        self.verbose = verbose
        self.precomputed_neighbours = None
        self.n_jobs = n_jobs
        self.workers = None
        if precompute:
            self.precomputed_neighbours = self.get_neighbour_indices(n_jobs=n_jobs)
        else:
            if n_jobs and os.cpu_count():
                # Negative worker counts wrap around to cpu core count, where -1 is one worker/core
                self.workers = ThreadPoolExecutor(
                    max_workers=n_jobs if n_jobs > 0 else os.cpu_count() + n_jobs + 1)

    def __getstate__(self):
        """ Return object serializable variable dict """

        state = dict(self.__dict__)
        state['index'] = None
        state['precomputed_neighbours'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        index = AnnoyIndex(self.index_dims, metric=self.metric)
        index.load(self.index_path)
        self.index = index

    def unload(self):
        """Unloads the index from disk, allowing other processes to read/write to the index file.
        After calling this, the index will no longer be usable from this instance."""
        self.index.unload()

    @classmethod
    def build(cls, X, path, k=150, metric='angular', search_k=-1, include_distances=False,
              ntrees=50, build_index_on_disk=True, precompute=False, verbose=1, n_jobs=-1):
        """Builds a new Annoy Index on input data *X*, then constructs and returns an
        AnnoyKnnMatrix object using the newly-built index."""

        _validate_knn_shape(X.shape[0], k)
        index = build_annoy_index(X, path, metric, ntrees, build_index_on_disk, verbose, n_jobs)
        return cls(index, X.shape[0], path, metric, k, search_k,
                   precompute, include_distances, verbose, n_jobs)

    @classmethod
    def load(cls, index_path, data_shape, k=150, metric='angular', search_k=-1,
             include_distances=False, precompute=False, verbose=1, n_jobs=-1):
        """Constructs and returns an AnnoyKnnMatrix object from an existing Annoy Index on disk."""

        _validate_knn_shape(data_shape[0], k)
        if not Path(index_path).exists():
            raise FileNotFoundError("Failed to load annoy index at '%s': "
                                    "file does not exist" % index_path)

        index = AnnoyIndex(data_shape[1], metric=metric)
        try:
            index.load(index_path)
        except IOError as err:
            raise IOError("Failed to load annoy index at '%s': file corrupt" % index_path) from err
        return cls(index, data_shape[0], index_path, metric, k, search_k,
                   precompute, include_distances, verbose, n_jobs)

    def __len__(self):
        """Number of rows in neighbour matrix"""
        return self.nrows

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

    def get_neighbour_indices(self, n_jobs=-1):
        """Retrieves neighbours for every row in parallel"""
        if self.precomputed_neighbours is not None:
            return self.precomputed_neighbours
        return extract_knn(self, verbose=self.verbose, n_jobs=n_jobs)

    def get_batch(self, idx_seq):
        """Returns a batch of neighbours based on the index sequence provided."""
        if self.workers is not None:
            return list(self.workers.map(self.__getitem__, idx_seq))
        return super().get_batch(idx_seq)

    def save(self, path):
        """Saves internal Annoy index to disk at given path."""
        self.index.save(path)

    def delete_index(self, parent=False):
        """Cleans up disk resources used by the index, rendering it unusable.
        First will `unload` the index, then recursively removes the files at index path.
        If parent is True, will recursively remove parent folder."""
        path = self.index_path if not parent else str(Path(self.index_path).parent)
        self.index.unload()
        shutil.rmtree(path)

def _validate_knn_shape(nrows, k):
    if k <= 0:
        raise ValueError('Invalid value of `%s` for k. k must be positive' %k)
    if k >= nrows:
        raise ValueError('''k value greater than or equal to num_rows
                            (k={}, rows={}). Lower k to a smaller
                            value.'''.format(k, nrows))

def build_annoy_index(X, path, metric='angular', ntrees=50, build_index_on_disk=True,
                      verbose=1, n_jobs=-1):
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

    batched_retrieval = hasattr(X, 'get_batch')

    index = AnnoyIndex(X.shape[1], metric=metric)
    if build_index_on_disk:
        index.on_disk_build(path)

    if not batched_retrieval:
        if issparse(X):
            for i in tqdm(range(X.shape[0]), disable=verbose < 1):
                vector = X[i].toarray()[0]
                index.add_item(i, vector)
        else:
            for i in tqdm(range(X.shape[0]), disable=verbose < 1):
                vector = X[i]
                index.add_item(i, vector)
    else:
        # Reusing n_jobs for batch_size here - separate config may be better
        batch_size = n_jobs if n_jobs > 0 else os.cpu_count() + n_jobs + 1
        for batch_start_idx in tqdm(range(0, X.shape[0], batch_size), disable=verbose < 1):
            batch_indices = range(batch_start_idx,
                                  min(batch_start_idx + batch_size, X.shape[0]))
            batch = X.get_batch(batch_indices)
            for idx, vector in zip(batch_indices, batch):
                index.add_item(idx, vector)

    try:
        index.build(ntrees, n_jobs=n_jobs)
    except Exception as e:
        raise Exception("Error building Annoy Index. Passing on_disk_build=False "
                        "may solve the issue, especially on Windows.") from e

    if not build_index_on_disk:
        index.save(path)
    return index

def extract_knn(knn_index, verbose=1, n_jobs=-1):
    """Starts multiple threads to retrieve nearest neighbours from a built index in parallel.

    :param `NeighbourMatrix` knn_index: Indexable neighbour index. When indexed, returns
        a list of neighbour indices for that row of the dataset it was built on.
    :param int verbose: Controls verbosity of output.
    """

    if verbose:
        print("Extracting KNN neighbours")

    nrows = len(knn_index)
    try:
        neighbour_matrix = np.empty((nrows, knn_index.k), dtype=get_uint_ctype(nrows))
    except (ValueError, MemoryError) as err:
        raise MemoryError("Unable to allocate memory for precomputed KNN matrix. "
                          "Set `precompute` to False or reduce the value for `k`.") from err

    worker_exception = None  # Halt signal
    def knn_worker(thread_index, data_indices):
        nonlocal worker_exception
        for i in range(data_indices[0], data_indices[1]):
            try:
                if worker_exception:
                    break
                neighbour_matrix[i] = knn_index[i]
                progress_counters[thread_index] += 1
            except Exception as e:
                worker_exception = e

    # Split up the indices and assign a thread for each chunk
    cpus_available = os.cpu_count() or 1
    n_jobs = n_jobs if n_jobs > 0 else cpus_available + n_jobs + 1
    chunk_size = max(1, nrows // n_jobs)
    data_split_indices = [(min_index, min(nrows, min_index + chunk_size))
                          for min_index in range(0, nrows, chunk_size)]

    progress_counters = [0 for _ in range(len(data_split_indices))]
    thread_pool = [
        Thread(
            target=knn_worker,
            kwargs={
                'thread_index': i,
                'data_indices': (min_index, max_index)
            }
        ) for i, (min_index, max_index) in enumerate(data_split_indices)
    ]

    try:
        for thread in thread_pool:
            thread.start()

        num_processed = 0
        # Poll for progress updates
        with tqdm(total=nrows, disable=verbose < 1) as pbar:
            while num_processed < nrows:
                # Raise worker errors
                if worker_exception:
                    raise worker_exception

                progress = sum(progress_counters) - num_processed
                num_processed += progress
                pbar.update(progress)
                time.sleep(0.1)

        return neighbour_matrix

    except BaseException as e:
        print('Error encountered. Halting KNN retrieval and cleaning up')
        # Signal worker threads to terminate
        worker_exception = e
        for thread in thread_pool:
            thread.join()
        raise
