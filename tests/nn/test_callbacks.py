import os
import tempfile
import pytest
import numpy as np
from ivis.nn.callbacks import ModelCheckpoint, EmbeddingsLogging, EmbeddingsImage, TensorBoardEmbeddingsImage
from ivis import Ivis


@pytest.fixture(scope='function')
def log_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_model_checkpoint(X, log_dir):
    filename = 'model-checkpoint_{}.ivis'
    n_epochs = 2
    model = Ivis(epochs=n_epochs, k=15, batch_size=16,
                 callbacks=[ModelCheckpoint(log_dir, filename, epoch_interval=1)])

    model.fit_transform(X)
    model_2 = Ivis()
    model_2.load_model(os.path.join(log_dir, filename.format(n_epochs)))

    # Test continuing training
    model_2.fit_transform(X)


def test_embeddings_logging(X, log_dir):
    filename = 'embeddings_{}.npy'
    n_epochs = 2
    model = Ivis(epochs=n_epochs, k=15, batch_size=16,
                 callbacks=[EmbeddingsLogging(X, log_dir, filename, epoch_interval=1)])

    y_pred = model.fit_transform(X)
    embeddings = np.load(os.path.join(log_dir, filename.format(n_epochs)))


def test_embeddings_image(X, Y, log_dir):
    filename = 'embeddings_{}.png'
    n_epochs = 2
    model = Ivis(epochs=n_epochs, k=15, batch_size=16,
                 callbacks=[EmbeddingsImage(X, Y, log_dir, filename, epoch_interval=1)])    

    model.fit_transform(X)
    assert os.path.exists(os.path.join(log_dir, filename.format(n_epochs)))


def test_embeddings_image(X, Y, log_dir):
    n_epochs = 2
    model = Ivis(epochs=n_epochs, k=15, batch_size=16,
                 callbacks=[TensorBoardEmbeddingsImage(X, Y, log_dir, epoch_interval=1)])    

    model.fit_transform(X)
    assert os.path.exists(os.path.join(log_dir, 'embeddings'))
