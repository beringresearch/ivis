from keras import backend as K

def triplet_euclidean_distance(vects):
    anchor, positive, negative = vects
    return _euclidean_distance(anchor, positive) - _euclidean_distance(anchor, negative)

def triplet_squared_euclidean_distance(vects):
    anchor, positive, negative = vects
    return K.square(_euclidean_distance(anchor, positive)) - K.square(_euclidean_distance(anchor, negative))

def triplet_normalized_euclidean_distance(vects):
    anchor, positive, negative = vects
    return K.l2_normalize(_euclidean_distance(anchor, positive), axis=1) - K.l2_normalize(_euclidean_distance(anchor, negative), axis=1)

def triplet_squared_normalized_euclidean_distance(vects):
    anchor, positive, negative = vects
    ap_norm_distance = K.l2_normalize(_euclidean_distance(anchor, positive), axis=1)
    an_norm_distance = K.l2_normalize(_euclidean_distance(anchor, negative), axis=1)
    return K.square(ap_norm_distance) - K.square(an_norm_distance)

def triplet_pn_distance(vects):
    anchor, positive, negative = vects
    
    anchor_positive_distance = _euclidean_distance(anchor, positive)
    anchor_negative_distance = _euclidean_distance(anchor, negative)
    positive_negative_distance = _euclidean_distance(positive, negative)
    
    minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=1, keepdims=True)
    
    return anchor_positive_distance - minimum_distance

def triplet_soft_pn_distance(vects):
    anchor, positive, negative = vects
    
    anchor_positive_distance = _euclidean_distance(anchor, positive)
    anchor_negative_distance = _euclidean_distance(anchor, negative)
    positive_negative_distance = _euclidean_distance(positive, negative)
    
    minimum_distance = K.min(K.concatenate([anchor_negative_distance, anchor_positive_distance]), axis=1, keepdims=True)
    
    ideal_distance = K.variable([0, 1])
    softmax = K.softmax(K.concatenate([anchor_positive_distance, minimum_distance]))
    return softmax - ideal_distance

def triplet_softmax_ratio_distance(vects):
    anchor, positive, negative = vects
    
    positive_distance = _euclidean_distance(anchor, positive)
    negative_distance = _euclidean_distance(anchor, negative)
    
    softmax = K.softmax(K.concatenate([positive_distance, negative_distance]))
    ideal_distance = K.variable([0, 1])
    return softmax - ideal_distance

def _euclidean_distance(x, y):
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))