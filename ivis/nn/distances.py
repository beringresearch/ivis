import tensorflow as tf
from tensorflow.keras import backend as K

def euclidean_distance(x, y):
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=-1, keepdims=True), K.epsilon()))


def manhattan_distance(x, y):
    return K.sum(K.abs(x - y), axis=-1, keepdims=True)


def chebyshev_distance(x, y):
    return K.max(K.abs(x - y), axis=-1, keepdims=True)


def cosine_distance(x, y):
    return K.sum(tf.nn.relu(1 + tf.keras.losses.cosine_similarity(x, y)))
