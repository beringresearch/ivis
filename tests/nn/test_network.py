from ivis.nn.network import get_base_networks, base_network, triplet_network

import keras
from keras.models import Model, Sequential
from keras.layers import Dense
import numpy as np

def test_base_networks():
    network_names = get_base_networks()
    input_shape = (4,)

    for name in network_names:
        model = base_network(name, input_shape)
        assert isinstance(model, Model)

def test_triplet_network():

    X = np.zeros(shape=(10, 5))
    embedding_dims = 3

    base_model = Sequential()
    base_model.add(Dense(8, input_shape=(X.shape[-1],)))

    model, _, _, _ = triplet_network(base_model, embedding_dims=embedding_dims, embedding_l2=0.1)
    encoder = model.layers[3]

    assert model.layers[3].output_shape == (None, 3)
    assert np.all(base_model.get_weights()[0] == encoder.get_weights()[0])
    assert np.all([isinstance(layer, keras.layers.InputLayer) for layer in model.layers[:3]])

    assert encoder.output_shape == (None, embedding_dims)

