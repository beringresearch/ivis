from keras import backend as K

def triplet_loss(margin=1):
    """ Traditional margin triplet loss."""
    def _triplet_loss_generic(y_true, y_pred):
        return K.mean(K.maximum(y_pred + margin, 0))
    return _triplet_loss_generic

def softmargin_loss(y_true, y_pred):
    return K.mean(K.softplus(y_pred))

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred))

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred))

if __name__ == '__main__':
    fun = triplet_loss(margin=0.1)