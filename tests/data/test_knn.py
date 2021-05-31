import tempfile
import os
import pytest
from annoy import AnnoyIndex
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np

from ivis.data.neighbour_retrieval import AnnoyKnnMatrix
from ivis.data.neighbour_retrieval.knn import build_annoy_index, extract_knn


@pytest.fixture(scope='function')
def annoy_index_file():
    with tempfile.TemporaryDirectory() as f:
        yield os.path.join(f, 'annoy.index')


def test_build_sparse_annoy_index(annoy_index_file):
    data = np.random.choice([0, 1], size=(10, 5))
    sparse_data = csr_matrix(data)

    index = build_annoy_index(sparse_data, annoy_index_file)
    assert os.path.exists(annoy_index_file)

    loaded_index = AnnoyIndex(5, metric='angular')
    loaded_index.load(annoy_index_file)

    assert index.f == loaded_index.f == 5
    assert index.get_n_items() == loaded_index.get_n_items() == 10
    assert index.get_nns_by_item(0, 5) == loaded_index.get_nns_by_item(0, 5)

    index.unload()
    loaded_index.unload()


def test_dense_annoy_index(annoy_index_file):
    data = np.random.choice([0, 1], size=(10, 5))
    index = build_annoy_index(data, annoy_index_file)
    assert os.path.exists(annoy_index_file)

    loaded_index = AnnoyIndex(5, metric='angular')
    loaded_index.load(annoy_index_file)

    assert index.f == loaded_index.f == 5
    assert index.get_n_items() == loaded_index.get_n_items() == 10
    assert index.get_nns_by_item(0, 5) == loaded_index.get_nns_by_item(0, 5)

    index.unload()
    loaded_index.unload()


def test_knn_retrieval():
    annoy_index_filepath = 'tests/data/.test-annoy-index.index'
    expected_neighbour_list = np.load('tests/data/test_knn_k3.npy')

    iris = datasets.load_iris()
    X = iris.data

    k = 3
    search_k = -1

    index = AnnoyKnnMatrix.load(annoy_index_filepath, X.shape, k=k, search_k=search_k)
    neighbour_list = extract_knn(index)

    assert np.all(expected_neighbour_list == neighbour_list)


def test_knn_matrix_construction_params(annoy_index_file):
    # Test too large k raises exception
    with pytest.raises(Exception):
        AnnoyKnnMatrix.build(np.zeros(shape=(4, 4)), annoy_index_file, k=4)
    with pytest.raises(Exception):
        AnnoyKnnMatrix.load(annoy_index_file, (4, 4), k=4)

    index = AnnoyKnnMatrix.build(np.zeros(shape=(4, 4)), annoy_index_file, k=2)
    loaded_index = AnnoyKnnMatrix.load(annoy_index_file, (4, 4), k=2)

    for original_row, loaded_row in zip(index, loaded_index):
        assert original_row == loaded_row
