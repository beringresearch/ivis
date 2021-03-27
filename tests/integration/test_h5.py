import os
import tempfile
import h5py
import numpy as np
import pytest
from ivis import Ivis


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
    with h5py.File(h5_filepath, 'r') as f:
        X_train = f['data']
        y_train = f['labels']

        # Train and transform with ivis
        model = Ivis(epochs=5, k=15, batch_size=16, precompute=False, build_index_on_disk=False)
        y_pred = model.fit_transform(X_train, shuffle_mode='batch')

        assert y_pred.shape[0] == len(X_train)
        assert y_pred.shape[1] == model.embedding_dims
