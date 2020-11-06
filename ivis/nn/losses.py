""" Triplet loss functions for training a siamese network with three subnetworks.
    All loss function variants are accessible through the `triplet_loss` function
    by specifying the distance as a string.
"""

import functools
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np

from .distances import euclidean_distance, manhattan_distance, chebyshev_distance, cosine_distance


loss_dict = {}
def register_loss(loss_fn=None, *, name=None):
    """Registers a class definition or class instance as a ivis loss function.
    A mapping will be created between the name and the loss function passed.
    If a class definition is provided, an instance will be created, passing the name
    as an argument.

    If no name is provided to this function, the name of the passed function will be used
    as a key.

    The loss function must have two parameters, (y_true, y_pred)
    and calculates the loss for a batch of triplet inputs (y_pred).
    y_pred is expected to be of shape: (3, batch_size, embedding_dims)."""

    if loss_fn is None:
        return functools.partial(register_loss, name=name)

    key = name or loss_fn.__name__
    if isinstance(loss_fn, type):
        loss_dict[key] = loss_fn(name=key)
    else:
        loss_dict[key] = loss_fn
    return loss_fn


def triplet_loss(distance='pn'):
    """Returns a created triplet loss function using provided hyperparameters.
    If passed a callable, just returns it."""
    if callable(distance):
        return distance
    try:
        loss_fn = loss_dict[distance]
        return loss_fn
    except KeyError:
        raise ValueError(f"Loss function {distance} not registered with ivis")


@register_loss(name='pn')
class EuclideanPnLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = euclidean_distance(anchor, positive)
        anchor_negative_distance = euclidean_distance(anchor, negative)
        positive_negative_distance = euclidean_distance(positive, negative)

        minimum_distance = K.min(
            [anchor_negative_distance, positive_negative_distance], axis=-1, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + self.margin, 0))


@register_loss(name='euclidean')
class EuclideanTripletLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)
        return K.mean(K.maximum(euclidean_distance(anchor, positive) - euclidean_distance(anchor, negative) + self.margin, 0))


@register_loss(name='manhattan')
class ManhattanTripletLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)
        return K.mean(K.maximum(manhattan_distance(anchor, positive) - manhattan_distance(anchor, negative) + self.margin, 0))


@register_loss(name='manhattan_pn')
class ManhattanPnLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = manhattan_distance(anchor, positive)
        anchor_negative_distance = manhattan_distance(anchor, negative)
        positive_negative_distance = manhattan_distance(positive, negative)

        minimum_distance = K.min(
            [anchor_negative_distance, positive_negative_distance], axis=-1, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + self.margin, 0))


@register_loss(name='chebyshev')
class ChebyshevTripletLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)
        return K.mean(K.maximum(chebyshev_distance(anchor, positive) - chebyshev_distance(anchor, negative) + self.margin, 0))


@register_loss(name='chebyshev_pn')
class ChebyshevPnLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = chebyshev_distance(anchor, positive)
        anchor_negative_distance = chebyshev_distance(anchor, negative)
        positive_negative_distance = chebyshev_distance(positive, negative)

        minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=-1, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + self.margin, 0))


@register_loss(name='cosine')
class CosineTripletLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)
        return K.mean(K.maximum(cosine_distance(anchor, positive) - cosine_distance(anchor, negative) + self.margin, 0))


@register_loss(name='cosine_pn')
class CosinePnLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = chebyshev_distance(anchor, positive)
        anchor_negative_distance = chebyshev_distance(anchor, negative)
        positive_negative_distance = chebyshev_distance(positive, negative)

        minimum_distance = K.min(tf.stack([anchor_negative_distance, positive_negative_distance]), axis=0, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + self.margin, 0))



@register_loss(name='softmax_ratio')
class EuclideanSoftmaxRatioLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)
        positive_distance = euclidean_distance(anchor, positive)
        negative_distance = euclidean_distance(anchor, negative)

        softmax = K.softmax(K.concatenate([positive_distance, negative_distance]))
        ideal_distance = K.variable([0, 1])
        return K.mean(K.maximum(softmax - ideal_distance, 0))


@register_loss(name='softmax_ratio_pn')
class EuclideanSoftmaxRatioPnLoss:
    def __init__(self, margin=1, name=None):
        self.margin = margin
        name = name or self.__class__.__name__
        self.__name__ = name
    def __call__(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = euclidean_distance(anchor, positive)
        anchor_negative_distance = euclidean_distance(anchor, negative)
        positive_negative_distance = euclidean_distance(positive, negative)

        minimum_distance = K.min(
            [anchor_negative_distance, positive_negative_distance], axis=-1, keepdims=True)

        softmax = K.softmax(K.concatenate([anchor_positive_distance, minimum_distance]))
        ideal_distance = K.variable([0, 1])
        return K.mean(K.maximum(softmax - ideal_distance, 0))


def semi_supervised_loss(loss_function):
    """Wraps the provided ivis supervised loss function to deal with the partially
    labeled data. Returns a new 'semi-supervised' loss function that masks the
    loss on examples where label information is missing.

    Missing labels are assumed to be marked with -1."""

    @functools.wraps(loss_function)
    def new_loss_function(y_true, y_pred):
        mask = tf.cast(~tf.math.equal(y_true, -1), tf.float32)
        y_true_pos = tf.nn.relu(y_true)
        loss = loss_function(y_true_pos, y_pred)
        masked_loss = loss * mask
        return masked_loss

    return new_loss_function

def is_hinge(supervised_loss):
    loss = keras.losses.get(supervised_loss)
    if loss in get_hinge_losses():
        return True
    return False


def get_hinge_losses():
    hinge_losses = set([keras.losses.hinge,
                        keras.losses.squared_hinge,
                        keras.losses.categorical_hinge])
    return hinge_losses


def get_categorical_losses():
    categorical_losses = set([keras.losses.sparse_categorical_crossentropy,
                              keras.losses.categorical_crossentropy,
                              keras.losses.categorical_hinge,
                              keras.losses.binary_crossentropy,
                              keras.losses.hinge,
                              keras.losses.squared_hinge])
    return categorical_losses


def get_multiclass_losses():
    multiclass_losses = set([keras.losses.sparse_categorical_crossentropy,
                             keras.losses.categorical_crossentropy,
                             keras.losses.categorical_hinge])
    return multiclass_losses


def is_categorical(supervised_loss):
    loss = keras.losses.get(supervised_loss)
    if loss in get_categorical_losses():
        return True
    return False


def is_multiclass(supervised_loss):
    loss = keras.losses.get(supervised_loss)
    if loss in get_multiclass_losses():
        return True
    return False


def validate_sparse_labels(Y):
    if not zero_indexed(Y):
        raise ValueError('Ensure that your labels are zero-indexed')
    if not consecutive_indexed(Y):
        raise ValueError('Ensure that your labels are indexed consecutively')


def zero_indexed(Y):
    if min(abs(Y)) != 0:
        return False
    return True


def consecutive_indexed(Y):
    """ Assumes that Y is zero-indexed. """
    n_classes = len(np.unique(Y[Y != np.array(-1)]))
    if max(Y) >= n_classes:
        return False
    return True
