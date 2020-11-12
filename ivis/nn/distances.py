import tensorflow as tf
from tensorflow.keras import backend as K

def euclidean_distance(x, y):
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=-1, keepdims=True), K.epsilon()))


def manhattan_distance(x, y):
    return K.sum(K.abs(x - y), axis=-1, keepdims=True)


def chebyshev_distance(x, y):
    return K.max(K.abs(x - y), axis=-1, keepdims=True)


def cosine_distance(x, y):
    return tf.math.reduce_sum(tf.nn.l2_normalize(x, axis=1) * tf.nn.l2_normalize(y, axis=1))
