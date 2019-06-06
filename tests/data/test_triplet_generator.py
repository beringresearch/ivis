from ivis.data.triplet_generators import generate_knn_triplets_from_neighbour_list
from sklearn import datasets

import numpy as np

def test_generate_knn_triplets_from_neighbour_list():
    neighbour_list = np.load('tests/data/test_knn_k4.npy')

    iris = datasets.load_iris()
    X = iris.data
    batch_size = 32

    data_generator = generate_knn_triplets_from_neighbour_list(X, neighbour_list, batch_size=batch_size)
    
    # Run generator thorugh one iteration of dataset and into the next
    for i in range((X.shape[0] // batch_size) + 2):
        batch = next(data_generator)

        # Check that everything is the expected shape
        assert isinstance(batch, tuple)
        assert len(batch) == 2

        assert len(batch[0]) == 3
        assert len(batch[1]) == batch_size
        assert batch[0][0].shape == (batch_size, X.shape[-1])


