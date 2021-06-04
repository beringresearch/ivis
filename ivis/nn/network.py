"""Contains helper functions for creating Siamese networks from base network architectures.
Additionally, provides several base network constructors as default options."""

from functools import partial
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, AlphaDropout, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from .losses import is_categorical, is_hinge, is_multiclass, validate_sparse_labels


base_networks = {}
def register_network(build_fn=None, *, name=None):
    """Registers a function that returns a tf.keras.models.Model as an ivis base network.
    A mapping will be created between the name and the network builder function passed.

    If no name is provided to this function, the name of the passed function will be used
    as a key.
    """

    if build_fn is None:
        return partial(register_network, name=name)

    key = name or build_fn.__name__
    base_networks[key] = build_fn
    return build_fn

def triplet_network(base_network, embedding_dims=2, embedding_l2=0.0):
    """ Creates a triplet Siamese Neural Network from a base_network.
    The base network will have an extra Dense layer of the requested embedding_dims added to
    the end if embedding_dims is not None.

    The outputs of the three network heads will be stacked into the shape:
    (3, batch_size, embedding_dims) unless embedding_dims is None (in which case the existing dims
    of last base_network layer will be used).

    Outputs: tuple(
        model: tf.keras.models.Model. The constructed triplet Siamese network
        embeddings: [tf.keras.layers.Dense, tf.keras.layers.Dense, tf.keras.layers.Dense].
        Results of applying the base_network to triplet inputs -
        anchor, positive and negative respectively.
    ) """

    def output_shape(shapes):
        shape1, _, _ = shapes
        return (3, shape1[0],)

    input_a = Input(shape=base_network.input_shape[1:])
    input_p = Input(shape=base_network.input_shape[1:])
    input_n = Input(shape=base_network.input_shape[1:])

    if embedding_dims is None:
        embeddings = base_network.output
    else:
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

    return model, (processed_a, processed_p, processed_n)


def base_network(model_name, input_shape):
    '''Return the defined base_network defined by the model_name string.
    '''
    try:
        return base_networks[model_name](input_shape)
    except KeyError:
        raise NotImplementedError(
            'Base network {} is not implemented'.format(model_name))


def get_base_networks():
    return list(base_networks.keys())

SeluDense = partial(Dense, activation='selu', kernel_initializer='lecun_normal')

@register_network(name='szubert')
def szubert_base_network(input_shape):
    '''A small, quick-to-train base network. The default for Ivis.'''
    return Sequential([
        SeluDense(128, input_shape=input_shape),
        AlphaDropout(0.1),
        SeluDense(128),
        AlphaDropout(0.1),
        SeluDense(128)
    ])

@register_network(name='hinton')
def hinton_base_network(input_shape):
    '''A base network inspired by the autoencoder architecture published in Hinton's paper
    'Reducing Dimensionality of Data with Neural Networks'
    (https://www.cs.toronto.edu/~hinton/science.pdf)'''
    return Sequential([
        SeluDense(2000, input_shape=input_shape),
        AlphaDropout(0.1),
        SeluDense(1000),
        AlphaDropout(0.1),
        SeluDense(500)
    ])

@register_network(name='maaten')
def maaten_base_network(input_shape):
    '''A base network inspired by the network architecture published in Maaten's t-SNE paper
    'Learning a Parametric Embedding by Preserving Local Structure'
    (https://lvdmaaten.github.io/publications/papers/AISTATS_2009.pdf)'''
    return Sequential([
        SeluDense(500, input_shape=input_shape),
        AlphaDropout(0.1),
        SeluDense(500),
        AlphaDropout(0.1),
        SeluDense(2000)
    ])


def build_supervised_layer(supervision_metric, Y, name='supervised'):
    """Constructs a dense layer suitable for the requested supervision metric and labels."""

    SupervisedDense = partial(Dense, name=name)
    SvmDense = partial(SupervisedDense, activation='linear', kernel_regularizer=regularizers.l2())

    # Regression
    if not is_categorical(supervision_metric):
        n_classes = Y.shape[-1] if len(Y.shape) > 1 else 1
        return SupervisedDense(n_classes, activation='linear')

    # Multiclass classification
    if is_multiclass(supervision_metric):
        if is_hinge(supervision_metric):
            # Multiclass Linear SVM layer
            n_classes = len(np.unique(Y, axis=0))
            return SvmDense(n_classes)
        # Softmax classifier
        validate_sparse_labels(Y)
        n_classes = len(np.unique(Y[Y != np.array(-1)]))
        return SupervisedDense(n_classes, activation='softmax')

    # Binary classification
    n_classes = Y.shape[-1] if len(Y.shape) > 1 else 1
    if is_hinge(supervision_metric):
        # Binary Linear SVM layer
        return SvmDense(n_classes)
    # Binary logistic classifier
    return SupervisedDense(n_classes, activation='sigmoid')
