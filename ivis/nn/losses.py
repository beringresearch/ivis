""" Triplet loss functions for training a siamese network with three subnetworks.
    All loss function variants are accessible through the `triplet_loss` function
    by specifying the distance as a string.
"""

from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np


loss_fn_builder_dict = {}
def register_loss_fn(loss_fn):
    """Registers a function as a ivis loss function builder. This builder function
    should return a loss function capable of dealing with triplet Tensors batches.

    This function allows users to register their own custom loss functions with ivis.

    Builder functions should have a parameter accepting keyword arguments **kwargs.
    The builder function should return a loss function with two parameters, (y_true, y_pred)
    which calculates the loss for a batch of triplet inputs.

    y_pred is expected to be of shape: (3, batch_size, embedding_dims)."""

    loss_fn_builder_dict[loss_fn.__qualname__] = loss_fn
    return loss_fn


def triplet_loss(distance='pn', margin=1):
    """Returns a created triplet loss function using provided hyperparameters"""
    loss_fn = loss_fn_builder_dict[distance](margin=margin)
    return loss_fn


def semi_supervised_loss(loss_function):
    """Wraps the provided ivis supervised loss function to deal with the partially
    labeled data. Returns a new 'semi-supervised' loss function that masks the
    loss on examples where label information is missing.

    Missing labels are assumed to be marked with -1."""

    def new_loss_function(y_true, y_pred):
        mask = tf.cast(~tf.math.equal(y_true, -1), tf.float32)
        y_true_pos = tf.nn.relu(y_true)
        loss = loss_function(y_true_pos, y_pred)
        masked_loss = loss * mask
        return masked_loss
    new_func = new_loss_function
    new_func.__name__ = loss_function.__name__
    return new_func


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


def _euclidean_distance(x, y):
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=-1, keepdims=True), K.epsilon()))


def _manhattan_distance(x, y):
    return K.sum(K.abs(x - y), axis=-1, keepdims=True)


def _chebyshev_distance(x, y):
    return K.max(K.abs(x - y), axis=-1, keepdims=True)


def _cosine_distance(x, y):
    return tf.math.reduce_sum(tf.nn.l2_normalize(x, axis=1) * tf.nn.l2_normalize(y, axis=1))


@register_loss_fn
def pn(margin=1):
    def _pn(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = _euclidean_distance(anchor, positive)
        anchor_negative_distance = _euclidean_distance(anchor, negative)
        positive_negative_distance = _euclidean_distance(positive, negative)

        minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=-1, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))

    return _pn

@register_loss_fn
def euclidean(margin=1):
    def _euclidean(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)        
        return K.mean(K.maximum(_euclidean_distance(anchor, positive) - _euclidean_distance(anchor, negative) + margin, 0))
    return _euclidean

@register_loss_fn
def manhattan(margin=1):
    def _manhattan(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)
        return K.mean(K.maximum(_manhattan_distance(anchor, positive) - _manhattan_distance(anchor, negative) + margin, 0))
    return _manhattan

@register_loss_fn
def manhattan_pn(margin=1):
    def _pn(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = _manhattan_distance(anchor, positive)
        anchor_negative_distance = _manhattan_distance(anchor, negative)
        positive_negative_distance = _manhattan_distance(positive, negative)

        minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=-1, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))

    return _pn

@register_loss_fn
def chebyshev(margin=1):
    def _chebyshev(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)  
        return K.mean(K.maximum(_chebyshev_distance(anchor, positive) - _chebyshev_distance(anchor, negative) + margin, 0))
    return _chebyshev

@register_loss_fn
def chebyshev_pn(margin=1):
    def _pn(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = _chebyshev_distance(anchor, positive)
        anchor_negative_distance = _chebyshev_distance(anchor, negative)
        positive_negative_distance = _chebyshev_distance(positive, negative)

        minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=-1, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))

    return _pn

@register_loss_fn
def cosine(margin=1):
    def _cosine(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)        
        return K.mean(K.maximum(_cosine_distance(anchor, positive) - _cosine_distance(anchor, negative) + margin, 0))
    return _cosine

@register_loss_fn
def cosine_pn(margin=1):
    def _pn(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = _cosine_distance(anchor, positive)
        anchor_negative_distance = _cosine_distance(anchor, negative)
        positive_negative_distance = _cosine_distance(positive, negative)

        minimum_distance = K.min(tf.stack([anchor_negative_distance, positive_negative_distance]), axis=0, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))

    return _pn

@register_loss_fn
def softmax_ratio(**_):
    def _softmax_ratio(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        positive_distance = _euclidean_distance(anchor, positive)
        negative_distance = _euclidean_distance(anchor, negative)

        softmax = K.softmax(K.concatenate([positive_distance, negative_distance]))
        ideal_distance = K.variable([0, 1])
        return K.mean(K.maximum(softmax - ideal_distance, 0))

    return _softmax_ratio

@register_loss_fn
def softmax_ratio_pn(**_):
    def _softmax_ratio_pn(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = _euclidean_distance(anchor, positive)
        anchor_negative_distance = _euclidean_distance(anchor, negative)
        positive_negative_distance = _euclidean_distance(positive, negative)

        minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=-1, keepdims=True)

        softmax = K.softmax(K.concatenate([anchor_positive_distance, minimum_distance]))
        ideal_distance = K.variable([0, 1])
        return K.mean(K.maximum(softmax - ideal_distance, 0))

    return _softmax_ratio_pn


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
