from ivis.data.knn import build_annoy_index, extract_knn

from annoy import AnnoyIndex
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets
import tempfile
import pytest
import os


@pytest.fixture(scope='function')
def annoy_index_file():
    _, filepath = tempfile.mkstemp('.index')
    yield filepath
    os.remove(filepath)

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
    expected_neighbour_list = np.load('tests/data/test_knn_k4.npy')
    
    iris = datasets.load_iris()
    X = iris.data
    
    k = 4
    search_k = -1
    neighbour_list = extract_knn(X, annoy_index_filepath, k=k, search_k=search_k)

    assert np.all(expected_neighbour_list == neighbour_list)