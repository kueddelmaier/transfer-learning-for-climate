import os
import tensorflow as tf

from climate_ae.data_generator import utils

# dimensionality of annotations
DANNO1 = 1000
DANNO2 = 1
DANNO3 = 1
# image dimensions
HEIGHT = 67
WIDTH = 101
DEPTH = 1
# data type
DTYPE = tf.float64


def train(directory):
    tfrecords_filename = 'train_temp_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2070'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test1(directory):
    tfrecords_filename = 'test_temp_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2070'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test2(directory):
    tfrecords_filename = 'holdout_temp_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2070'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds
