import numpy as np
import tensorflow as tf
import pdb


def parse_dataset(example_proto, img_size_h, img_size_w, img_size_d, dim_anno1, dtype_img=tf.float64):

    features = {
        'inputs': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'annotations': tf.io.FixedLenFeature(shape=[dim_anno1], dtype=tf.float32),
        'year': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        'month': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        'day': tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, features=features)
    image = tf.io.decode_raw(parsed_features["inputs"], dtype_img)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [img_size_h, img_size_w, img_size_d])
    annotations = parsed_features["annotations"]
    year = parsed_features["year"]
    month = parsed_features["month"]
    day = parsed_features["day"]

    return image, annotations, year, month, day


def climate_dataset(file_path, height, width, depth, dim_anno1, dtype):
    dataset = tf.data.TFRecordDataset(file_path,)
    dataset = dataset.map(lambda x: parse_dataset(x, height, width, depth, 
        dim_anno1, dtype_img=dtype))

    return dataset