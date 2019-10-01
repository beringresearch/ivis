import h5py
from ivis import Ivis
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import HDF5Matrix
import numpy as np
import pytest
import os
import tempfile


@pytest.fixture(scope='function')
def h5_filepath():
    _, filepath = tempfile.mkstemp('.h5')
    yield filepath
    os.remove(filepath)


def create_random_dataset(path, rows, dims):
    X = np.random.randn(rows, dims).astype('float32')
    y = np.random.randint(0, 2, size=(rows, 1))
    with h5py.File(path, 'w') as f:
        X_dataset = f.create_dataset('data', (rows, dims), dtype='f')
        X_dataset[:] = X
        y_dataset = f.create_dataset('labels', (len(X_dataset), 1), dtype='i')
        y_dataset[:] = y


def test_h5_file(h5_filepath):
    rows, dims = 258, 32
    create_random_dataset(h5_filepath, rows, dims)

    # Load data
    test_index = rows // 5
    X_train = HDF5Matrix(h5_filepath, 'data', start=0, end=test_index)
    y_train = HDF5Matrix(h5_filepath, 'labels', start=0, end=test_index)

    X_test = HDF5Matrix(h5_filepath, 'data', start=test_index, end=rows)
    y_test = HDF5Matrix(h5_filepath, 'labels', start=test_index, end=rows)

    # Train and transform with ivis
    ivis_iris = Ivis(epochs=5, k=15, batch_size=16)

    y_pred_iris = ivis_iris.fit_transform(X_train, shuffle_mode='batch')
    y_pred = ivis_iris.transform(X_test)

    assert y_pred.shape[0] == len(X_test)
    assert y_pred.shape[1] == ivis_iris.embedding_dims
