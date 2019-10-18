""" Creates a Siamese Dense Neural Network with three subnetworks """

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, AlphaDropout, Lambda
from tensorflow.keras import backend as K

from tensorflow.keras.regularizers import l2


def triplet_network(base_network, embedding_dims=2, embedding_l2=0.0):
    def output_shape(shapes):
        shape1, shape2, shape3 = shapes
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
    if model_name == 'szubert':
        return szubert_base_network(input_shape)
    elif model_name == 'hinton':
        return hinton_base_network(input_shape)
    elif model_name == 'maaten':
        return maaten_base_network(input_shape)

    raise NotImplementedError(
        'Base network {} is not implemented'.format(model_name))


def get_base_networks():
    return ['szubert', 'hinton', 'maaten']


def szubert_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
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
    '''Base network to be shared (eq. to feature extraction).
    '''
    inputs = Input(shape=input_shape)
    x = Dense(2000, activation='selu',
              kernel_initializer='lecun_normal')(inputs)
    x = AlphaDropout(0.1)(x)
    x = Dense(1000, activation='selu', kernel_initializer='lecun_normal')(x)
    x = AlphaDropout(0.1)(x)
    x = Dense(500, activation='selu', kernel_initializer='lecun_normal')(x)
    return Model(inputs, x)


def maaten_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    inputs = Input(shape=input_shape)
    x = Dense(500, activation='selu',
              kernel_initializer='lecun_normal')(inputs)
    x = AlphaDropout(0.1)(x)
    x = Dense(500, activation='selu', kernel_initializer='lecun_normal')(x)
    x = AlphaDropout(0.1)(x)
    x = Dense(2000, activation='selu', kernel_initializer='lecun_normal')(x)
    return Model(inputs, x)
