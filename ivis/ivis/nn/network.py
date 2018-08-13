""" Trains a DNN using Triplet Loss.
"""
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, AlphaDropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

from keras import regularizers, constraints
from keras.layers.core import Dense

def build_network(base_network, distance_function):

    def output_shape(shapes):
        shape1, shape2, shape3 = shapes
        return (shape1[0], 1)

    input_a = Input(shape=base_network.input_shape[1:])
    input_p = Input(shape=base_network.input_shape[1:])
    input_n = Input(shape=base_network.input_shape[1:])

    processed_a = base_network(input_a)
    processed_p = base_network(input_p)
    processed_n = base_network(input_n)

    distance = Lambda(distance_function,
                      output_shape=output_shape)([processed_a, processed_p, processed_n])

    model = Model([input_a, input_p, input_n], distance)
    return model

def selu_base_network(input_shape, embedding_l2=0.0):
    '''Base network to be shared (eq. to feature extraction).
    '''
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='selu', kernel_initializer='lecun_normal')(inputs)
    x = AlphaDropout(0.1)(x)
    x = Dense(128, activation='selu', kernel_initializer='lecun_normal')(x)
    x = AlphaDropout(0.1)(x)
    x = Dense(128, activation='selu', kernel_initializer='lecun_normal')(x)
    x = Dense(2, kernel_regularizer=regularizers.l2(embedding_l2))(x)
    return Model(inputs, x)

def relu_base_network(input_shape, embedding_l2=0.0):
    '''Base network to be shared (eq. to feature extraction).
    '''
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(inputs)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(2, kernel_regularizer=regularizers.l2(embedding_l2))(x)
    return Model(inputs, x)

def tanh_base_network(input_shape, embedding_l2=0.0):
    '''Base network to be shared (eq. to feature extraction).
    '''
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='tanh', kernel_initializer='glorot_uniform')(inputs)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='tanh', kernel_initializer='glorot_uniform')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='tanh', kernel_initializer='glorot_uniform')(x)
    x = Dense(2, kernel_regularizer=regularizers.l2(embedding_l2))(x)
    return Model(inputs, x)

if __name__ == "__main__":
    from losses import triplet_loss
    from distance_metrics import triplet_euclidean_distance
    input_shape=(32,)
    model = build_network(selu_base_network(input_shape, embedding_l2=0.01), triplet_euclidean_distance)
    model.compile(optimizer='adam', loss=triplet_loss(margin=0.1))