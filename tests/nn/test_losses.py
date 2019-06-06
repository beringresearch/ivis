from ivis.nn import losses
from ivis.nn.losses import triplet_loss, get_loss_functions

from keras import backend as K
import numpy as np
import tensorflow as tf


def test_loss_function_call():
    margin = 2

    loss_dict = get_loss_functions(margin=margin)
    
    for loss_name in loss_dict.keys():
        loss_function = triplet_loss(distance=loss_name, margin=margin)
        assert loss_function.__name__ == loss_name
        
