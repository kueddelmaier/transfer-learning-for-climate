import rpy2.robjects as robjects
import numpy as np
import random

import os
import tensorflow as tf
import argparse


# helper functions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float32_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature_list(value):
    return tf.train.Feature(float_list=tf.train.Int64List(value=value))


def convert_rda_to_tf_records(file, rda_folder, tfrecords_folder, sub):

    print('processing: ' + file )

    save_name = file.split('.')[0] + '.tfrecords'
    
    robjects.r['load'](os.path.join(rda_folder, file))
    Z = np.array(robjects.r['psl_Z_'+sub])

    dates = np.array(robjects.r['dates_'+sub])
    years_int = [int(d[1:5]) for d in dates]
    months_int = [int(d[6:8]) for d in dates]
    days_int = [int(d[9:11]) for d in dates]


    years_int = np.expand_dims(np.array(years_int), axis=1)
    months_int = np.expand_dims(np.array(months_int), axis=1)
    days_int = np.expand_dims(np.array(days_int), axis=1)

    images = np.array(robjects.r['prec_mat_sqrt_'+sub])

    filename = os.path.join(tfrecords_folder, save_name)
    writer = tf.io.TFRecordWriter(filename)
    print(images.shape)
    print(years_int.shape)
    print('----------')
    for i in range(images.shape[2]):
        # image
        img_t = (images[:, :, i])
        img = _bytes_feature(img_t.tostring())
        
        # annotation
        anno = _float32_feature_list(Z[i,:].astype(np.float32))
        year = _float32_feature_list(years_int[i,:].astype(np.float32))
        month = _float32_feature_list(months_int[i,:].astype(np.float32))
        day = _float32_feature_list(days_int[i,:].astype(np.float32))
    
        example = tf.train.Example(features=
        tf.train.Features(feature={
            'inputs': img,
            'annotations': anno,
            'year': year,
            'month': month,
            'day': day 
        }))

        writer.write(example.SerializeToString())

    writer.close()
        

def main():

    #folder = "prec_psl_train_mixed11_holdout_all_reanalysis_from_1955_id_QLFHS6563Q_date_2022_06_09__15_03_detrend_TRUE_DEBUG_FALSE_new_rda"
    #folder = "prec_psl_train_mixed11_holdout_all_reanalysis_from_1955_date_2022_08_17__11_24_id_MGXXW5859Y_detrend_TRUE_DEBUG_FALSE"
    folder = "prec_psl_train_mixed11_holdout_all_reanalysis_from_1955_date_2023_10_13__02_06_id_TOEZZ3370E_detrend_TRUE_DEBUG_FALSE"
    dir = "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/preprocessed_data/MIX"
    folder_path = os.path.join(dir, folder)
    rda_folder = os.path.join(folder_path, 'rda_data')
    assert os.path.isdir(rda_folder), "RDA folder does not exist"

    tfrecords_folder = os.path.join(folder_path, 'tfrecords_data')
    os.makedirs(tfrecords_folder, exist_ok=True)

    for file in os.listdir(rda_folder):
        
        if 'train' in file:
            convert_rda_to_tf_records(file, rda_folder, tfrecords_folder, sub = 'tr')

        if 'test' in file:
            convert_rda_to_tf_records(file, rda_folder, tfrecords_folder, sub = 'te')

        if 'holdout' in file:
            convert_rda_to_tf_records(file, rda_folder, tfrecords_folder, sub = 'ho')


if __name__ == '__main__':
    main()


