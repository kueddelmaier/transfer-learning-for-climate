import os
import tensorflow as tf

from climate_ae.data_generator import utils

# dimensionality of annotations
DANNO1 = 1000

# image dimensions
HEIGHT = 66
WIDTH = 100
DEPTH = 1
# data type
DTYPE = tf.float64

def return_prec_dataset(filepath):

    ds = utils.climate_dataset(filepath, HEIGHT, WIDTH, DEPTH, DANNO1, DTYPE)
    return ds


