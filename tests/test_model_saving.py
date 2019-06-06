import pytest
from ivis import Ivis
import os
import shutil
from sklearn import datasets
import numpy as np
import tempfile

@pytest.fixture(scope='function')
def model_filepath():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_filepath = os.path.join(temp_dir, 'test.ivis.model.saving.ivis')
        yield model_filepath

def test_ivis_model_saving(model_filepath):
    model = Ivis(k=15, batch_size=16, n_epochs_without_progress=5)
    iris = datasets.load_iris()
    X = iris.data

    model.fit(X)
    model.save_model(model_filepath)

    model_2 = Ivis()
    model_2.load_model(model_filepath)

    # Check that model predictions are same
    assert np.all(model.transform(X) == model_2.transform(X))
    assert model.__getstate__() == model_2.__getstate__() # Serializable dict eles same

    # Check all weights are the same
    for model_layer, model_2_layer in zip(model.encoder.layers, model_2.encoder.layers):
        model_layer_weights = model_layer.get_weights()
        model_2_layer_weights = model_2_layer.get_weights()
        for i in range(len(model_layer_weights)):
            assert np.all(model_layer_weights[i] == model_2_layer_weights[i])

    # Check optimizer weights are the same
    for model_optimizer_weights, model_2_optimizer_weights in zip(model.model_.optimizer.get_weights(), model_2.model_.optimizer.get_weights()):            
        assert np.all(model_optimizer_weights == model_2_optimizer_weights)

    # Check that trying to save over an existing folder raises an Exception
    with pytest.raises(FileExistsError) as exception_info:
        model.save_model(model_filepath)
    assert isinstance(exception_info.value, FileExistsError)

