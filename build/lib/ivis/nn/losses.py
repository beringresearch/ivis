""" Triplet loss functions for training a siamese network with three subnetworks.
    All loss function variants are accessible through the `triplet_loss` function by specifying the distance as a string.
"""

from keras import backend as K
import tensorflow as tf

def triplet_loss(distance='pn', margin=1):

    losses = get_loss_functions(margin=margin)

    loss_function = losses[distance]
    loss_function.__name__ = distance

    return loss_function

def get_loss_functions(margin=1):
    losses = {
        'pn' : pn_loss(margin=margin),
        'euclidean' : euclidean_loss(margin=margin),
        'softmax_ratio' : softmax_ratio,
        'softmax_ratio_pn' : softmax_ratio_pn,
        'manhattan': manhattan_loss(margin=margin),
        'manhattan_pn': manhattan_pn_loss(margin=margin),
        'chebyshev': chebyshev_loss(margin=margin),
        'chebyshev_pn': chebyshev_pn_loss(margin=margin)
    }
    return losses

def _euclidean_distance(x, y):
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def _manhattan_distance(x, y):
    return K.sum(K.abs(x - y), axis=1, keepdims=True)

def _chebyshev_distance(x, y):
    return K.max(K.abs(x - y), axis=1, keepdims=True)

def pn_loss(margin=1):
    def _pn_loss(y_true, y_pred):    
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = _euclidean_distance(anchor, positive)
        anchor_negative_distance = _euclidean_distance(anchor, negative)
        positive_negative_distance = _euclidean_distance(positive, negative)

        minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=1, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))
    
    return _pn_loss 

def euclidean_loss(margin=1):
    def _euclidean_loss(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)        
        return K.mean(K.maximum(_euclidean_distance(anchor, positive) - _euclidean_distance(anchor, negative) + margin, 0))
    return _euclidean_loss

def manhattan_loss(margin=1):
    def _manhattan_loss(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)        
        return K.mean(K.maximum(_manhattan_distance(anchor, positive) - _manhattan_distance(anchor, negative) + margin, 0))
    return _manhattan_loss

def manhattan_pn_loss(margin=1):
    def _pn_loss(y_true, y_pred):    
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = _manhattan_distance(anchor, positive)
        anchor_negative_distance = _manhattan_distance(anchor, negative)
        positive_negative_distance = _manhattan_distance(positive, negative)

        minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=1, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))
    
    return _pn_loss

def chebyshev_loss(margin=1):
    def _chebyshev_loss(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)  
        return K.mean(K.maximum(_chebyshev_distance(anchor, positive) - _chebyshev_distance(anchor, negative) + margin, 0))
    return _chebyshev_loss

def chebyshev_pn_loss(margin=1):
    def _pn_loss(y_true, y_pred):
        anchor, positive, negative = tf.unstack(y_pred)

        anchor_positive_distance = _chebyshev_distance(anchor, positive)
        anchor_negative_distance = _chebyshev_distance(anchor, negative)
        positive_negative_distance = _chebyshev_distance(positive, negative)

        minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=1, keepdims=True)

        return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))
    
    return _pn_loss

def softmax_ratio(y_true, y_pred):
    anchor, positive, negative = tf.unstack(y_pred)
    
    positive_distance = _euclidean_distance(anchor, positive)
    negative_distance = _euclidean_distance(anchor, negative)

    softmax = K.softmax(K.concatenate([positive_distance, negative_distance]))
    ideal_distance = K.variable([0, 1])
    return K.mean(K.maximum( softmax - ideal_distance, 0))

def softmax_ratio_pn(y_true, y_pred):
    anchor, positive, negative = tf.unstack(y_pred)
    
    anchor_positive_distance = _euclidean_distance(anchor, positive)
    anchor_negative_distance = _euclidean_distance(anchor, negative)
    positive_negative_distance = _euclidean_distance(positive, negative)
    
    minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=1, keepdims=True)

    softmax = K.softmax(K.concatenate([anchor_positive_distance, minimum_distance]))
    ideal_distance = K.variable([0, 1])
    return K.mean(K.maximum( softmax - ideal_distance, 0))
