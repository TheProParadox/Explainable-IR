import tensorflow as tf
from keras.losses import *
from keras.layers import Lambda
from tensorflow.python.keras.utils import deserialize_keras_object
from keras import layers, backend as K


def rank_hinge_loss(y_true, y_pred):
    y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
    loss = K.maximum(0., 1. + y_neg - y_pos)
    return K.mean(loss)


def serialize(rank_loss):
    return rank_loss.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


