import numpy as np
import pytest

from sklearn import datasets

from ivis.data.triplet_generators import generator_from_index
from ivis.data.triplet_generators import KnnTripletGenerator
from ivis.data.triplet_generators import AnnoyTripletGenerator


def test_KnnTripletGenerator():
    neighbour_list = np.load('tests/data/test_knn_k4.npy')

    iris = datasets.load_iris()
    X = iris.data
    batch_size = 32

    data_generator = KnnTripletGenerator(X, neighbour_list,
                                         batch_size=batch_size)

    # Run generator thorugh one iteration of dataset and into the next
    for i in range((X.shape[0] // batch_size) + 1):
        batch = data_generator.__getitem__(i)

        # Check that everything is the expected shape
        assert isinstance(batch, tuple)
        assert len(batch) == 2

        assert len(batch[0]) == 3
        assert len(batch[1]) <= batch_size
        assert batch[0][0].shape[-1] == X.shape[-1]


def test_AnnoyTripletGenerator():
    neighbour_list = np.load('tests/data/test_knn_k4.npy')

    iris = datasets.load_iris()
    X = iris.data
    batch_size = 32

    data_generator = KnnTripletGenerator(X, neighbour_list,
                                         batch_size=batch_size)

    # Run generator thorugh one iteration of dataset and into the next
    for i in range((X.shape[0] // batch_size) + 1):
        batch = data_generator.__getitem__(i)

        # Check that everything is the expected shape
        assert isinstance(batch, tuple)
        assert len(batch) == 2

        assert len(batch[0]) == 3
        assert len(batch[1]) <= batch_size
        assert batch[0][0].shape[-1] == X.shape[-1]


def test_generator_from_index():
        # Test too large k raises exception
        with pytest.raises(Exception):
                generator_from_index(np.zeros(shape=(4, 5)), 
                                     'placeholder_path.index',
                                     k=10,
                                     batch_size=2,
                                     search_k=1,
                                     precompute=False,
                                     verbose=0)
        # Test too large batch_size raises exception
        with pytest.raises(Exception):
                generator_from_index(np.zeros(shape=(4, 5)), 
                                     'placeholder_path.index',
                                     k=2,
                                     batch_size=8,
                                     search_k=1,
                                     precompute=False,
                                     verbose=0)
