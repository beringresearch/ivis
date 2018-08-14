""" Creates a Siamese Dense Neural Network with three subnetworks """

from .losses import triplet_loss

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, AlphaDropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

from keras import regularizers, constraints
from keras.layers.core import Dense


def build_network(base_network):
    def output_shape(shapes):
        shape1, shape2, shape3 = shapes
        return (3, shape1[0],)

    input_a = Input(shape=base_network.input_shape[1:])
    input_p = Input(shape=base_network.input_shape[1:])
    input_n = Input(shape=base_network.input_shape[1:])

    processed_a = base_network(input_a)
    processed_p = base_network(input_p)
    processed_n = base_network(input_n)

    triplet = Lambda(K.stack, output_shape=output_shape)([processed_a, processed_p, processed_n])
    model = Model([input_a, input_p, input_n], triplet)

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

if __name__ == "__main__":
    input_shape=(32,)
    model = build_network(selu_base_network(input_shape, embedding_l2=0.01))
    model.compile(optimizer='adam', loss=triplet_loss('pn', margin=0.1))