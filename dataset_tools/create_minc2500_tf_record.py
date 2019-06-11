"""
Convert MINC-2500 dataset to TFRecord for classification.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import random
import numpy as np

import PIL.Image
import tensorflow as tf

from dataset_tools import dataset_util


flags = tf.app.flags
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/minc-2500',
                    'Root Directory to raw minc-2500 dataset.')
# make tfrecord by one of labels
flags.DEFINE_string('output_path',
                    '/home/ace19/dl_data/minc-2500/validate.record',
                    'Path to output TFRecord')
flags.DEFINE_string('label_map_path',
                    '/home/ace19/dl_data/minc-2500/labels/validate3.txt',
                    'Path to label map')
flags.DEFINE_string('dataset_category',
                    'train',
                    'dataset category, train|validate|test')

FLAGS = flags.FLAGS


def get_label_map_dict(label_map_path, label_to_index):
    label_map_dict = {}
    image_map_dict = {}
    with open(label_map_path, 'r') as reader:
        for line in reader:
            fields = line.strip().split('/')
            try:
                label_map_dict[fields[2]] = label_to_index[fields[1]]
                image_map_dict[fields[2]] = os.path.join(FLAGS.dataset_dir, line.strip())
            except KeyError:
                continue

    # random shuffle
    l = list(map(tuple, label_map_dict.items()))
    random.shuffle(l)
    label_map_dict = dict(l)

    return label_map_dict, image_map_dict


def dict_to_tf_example(image_name,
                       label,
                       image_map_dict=None):
    """
    Args:
      image: a single image name
      label_map_dict: A map from string label names to integers ids.
      image_map_dict: A map from string label names to image path

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by image is not a valid PNG
    """
    full_image_path = image_map_dict[image_name]
    with tf.gfile.GFile(full_image_path, 'rb') as fid:
        encoded = fid.read()
    encoded_io = io.BytesIO(encoded)
    image = PIL.Image.open(encoded_io)
    width, height = image.size
    format = image.format
    image_stat = PIL.ImageStat.Stat(image)
    mean = image_stat.mean
    std = image_stat.stddev
    if image.format != 'JPEG':
        raise ValueError('Image format not jpg')
    key = hashlib.sha256(encoded).hexdigest()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_name.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded),
        'image/format': dataset_util.bytes_feature(format.encode('utf8')),
        'image/label': dataset_util.int64_feature(label),
        # 'image/text': dataset_util.bytes_feature('label_text'.encode('utf8'))
        'image/mean': dataset_util.float_list_feature(mean),
        'image/std': dataset_util.float_list_feature(std)
    }))
    return example


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path, options=options)

    source = os.path.join(FLAGS.dataset_dir, 'images')
    dataset_lst = os.listdir(source)
    dataset_lst.sort()
    label_to_index = {}
    for i, cls in enumerate(dataset_lst):
        cls_path = os.path.join(source, cls)
        if os.path.isdir(cls_path):
            label_to_index[cls] = i

    label_map_dict = None
    image_map_dict = None
    if FLAGS.label_map_path:
        label_map_dict, image_map_dict = \
            get_label_map_dict(FLAGS.label_map_path, label_to_index)

    idx = 0
    tf.logging.info('Reading from minc-2500 dataset.')
    num_label_map_dict = len(label_map_dict.keys())
    for image, label in label_map_dict.items():
        if idx % 100 == 0:
            tf.logging.info('On image %d of %d', idx, num_label_map_dict)
        tf_example = dict_to_tf_example(image, label, image_map_dict)
        writer.write(tf_example.SerializeToString())
        idx += 1

    writer.close()


if __name__ == '__main__':
    tf.app.run()
