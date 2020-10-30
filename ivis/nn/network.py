"""Contains helper functions for creating Siamese networks from base network architectures.
Additionally, provides several base network constructors as default options."""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, AlphaDropout, Lambda
from tensorflow.keras import backend as K

from tensorflow.keras.regularizers import l2


def triplet_network(base_network, embedding_dims=2, embedding_l2=0.0):
    """ Creates a triplet Siamese Neural Network from a base_network.
    The base network will have an extra Dense layer of the requested embedding_dims added to
    the end if embedding_dims is not None.

    The outputs of the three network heads will be stacked into the shape:
    (3, batch_size, embedding_dims) unless embedding_dims is None (in which case the existing dims
    of last base_network layer will be used).

    Outputs: tuple(
        model: tf.keras.models.Model. The constructed triplet Siamese network
        processed_a: tf.keras.layers.Dense. Result of applying the base_network to anchor input.
        processed_p: tf.keras.layers.Dense. Result of applying the base_network to positive input.
        processed_n: tf.keras.layers.Dense. Result of applying the base_network to negative input.
    ) """

    def output_shape(shapes):
        shape1, _, _ = shapes
        return (3, shape1[0],)

    input_a = Input(shape=base_network.input_shape[1:])
    input_p = Input(shape=base_network.input_shape[1:])
    input_n = Input(shape=base_network.input_shape[1:])

    embeddings = Dense(embedding_dims,
                       kernel_regularizer=l2(embedding_l2))(base_network.output)
    network = Model(base_network.input, embeddings)

    processed_a = network(input_a)
    processed_p = network(input_p)
    processed_n = network(input_n)

    triplet = Lambda(K.stack,
                     output_shape=output_shape,
                     name='stacked_triplets')([processed_a,
                                               processed_p,
                                               processed_n],)
    model = Model([input_a, input_p, input_n], triplet)

    return model, processed_a, processed_p, processed_n


def base_network(model_name, input_shape):
    '''Return the defined base_network defined by the model_name string.
    '''
    # Select network to return in if statements to avoid running and
    # constructing unnecessary models as would occur in a dict.
    if model_name == 'szubert':
        return szubert_base_network(input_shape)
    if model_name == 'hinton':
        return hinton_base_network(input_shape)
    if model_name == 'maaten':
        return maaten_base_network(input_shape)

    raise NotImplementedError(
        'Base network {} is not implemented'.format(model_name))


def get_base_networks():
    return ['szubert', 'hinton', 'maaten']


def szubert_base_network(input_shape):
    '''A small, quick-to-train base network. The default for Ivis.'''
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='selu',
              kernel_initializer='lecun_normal')(inputs)
    x = AlphaDropout(0.1)(x)
    x = Dense(128, activation='selu',
              kernel_initializer='lecun_normal')(x)
    x = AlphaDropout(0.1)(x)
    x = Dense(128, activation='selu', kernel_initializer='lecun_normal')(x)
    return Model(inputs, x)


def hinton_base_network(input_shape):
    '''A base network inspired by the autoencoder architecture published in Hinton's paper
    'Reducing Dimensionality of Data with Neural Networks' (https://www.cs.toronto.edu/~hinton/science.pdf)'''
    inputs = Input(shape=input_shape)
    x = Dense(2000, activation='selu',
              kernel_initializer='lecun_normal')(inputs)
    x = AlphaDropout(0.1)(x)
    x = Dense(1000, activation='selu', kernel_initializer='lecun_normal')(x)
    x = AlphaDropout(0.1)(x)
    x = Dense(500, activation='selu', kernel_initializer='lecun_normal')(x)
    return Model(inputs, x)


def maaten_base_network(input_shape):
    '''A base network inspired by the network architecture published in Maaten's t-SNE paper
    'Learning a Parametric Embedding by Preserving Local Structure'
    (https://lvdmaaten.github.io/publications/papers/AISTATS_2009.pdf)'''
    inputs = Input(shape=input_shape)
    x = Dense(500, activation='selu',
              kernel_initializer='lecun_normal')(inputs)
    x = AlphaDropout(0.1)(x)
    x = Dense(500, activation='selu', kernel_initializer='lecun_normal')(x)
    x = AlphaDropout(0.1)(x)
    x = Dense(2000, activation='selu', kernel_initializer='lecun_normal')(x)
    return Model(inputs, x)
