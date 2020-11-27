import numpy as np
import pytest

from sklearn import datasets

from ivis.data.triplet_generators import generator_from_neighbour_matrix
from ivis.data.triplet_generators import UnsupervisedTripletGenerator


def test_UnsupervisedTripletGenerator():
    neighbour_list = np.load('tests/data/test_knn_k3.npy')

    iris = datasets.load_iris()
    X = iris.data
    batch_size = 32

    data_generator = UnsupervisedTripletGenerator(X, neighbour_list,
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


def test_generator_from_neighbour_matrix():
    # Test too large batch_size raises exception
    with pytest.raises(Exception):
        generator_from_neighbour_matrix(np.zeros(shape=(4, 4)),
                                        None, 
                                        np.zeros(shape=(4, 2)), batch_size=32)
    with pytest.raises(Exception):
        generator_from_neighbour_matrix(np.zeros(shape=(4, 4)),
                                        np.zeros(shape=(4,)), 
                                        np.zeros(shape=(4, 2)), batch_size=32)

    # Test with valid hyperparameters
    unsupervised_gen = generator_from_neighbour_matrix(np.zeros(shape=(4, 4)),
                                          None, 
                                          np.zeros(shape=(4, 2)), batch_size=2)
    for triplets, labels in unsupervised_gen:
        assert isinstance(triplets, tuple)
        assert(len(triplets) == 3)
        for triplet in triplets:
            assert(triplet.shape == (2, 4))

    supervised_gen = generator_from_neighbour_matrix(np.zeros(shape=(4, 4)),
                                          np.ones(shape=(4,)), 
                                          np.zeros(shape=(4, 2)), batch_size=2)
    for triplets, labels in supervised_gen:
        assert isinstance(triplets, tuple)
        assert(len(triplets) == 3)
        for triplet in triplets:
            assert(triplet.shape == (2, 4))
        assert(isinstance(labels, tuple))
        assert(np.all(np.unique(labels) == 1))
