from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import random


MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]


class Dataset(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Handles loading, partitioning, and preparing training data.
    """

    def __init__(self, tfrecord_path, height, width, num_epochs, batch_size):
        self.resize_h = height
        self.resize_w = width

        dataset = tf.data.TFRecordDataset(tfrecord_path,
                                          compression_type='GZIP',
                                          num_parallel_reads=batch_size * 4)
        # dataset = dataset.map(self._parse_func, num_parallel_calls=8)
        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(self.decode, num_parallel_calls=8)
        dataset = dataset.map(self.augment, num_parallel_calls=8)
        dataset = dataset.map(self.normalize, num_parallel_calls=8)

        # Prefetches a batch at a time to smooth out the time taken to load input
        # files for shuffling and processing.
        dataset = dataset.prefetch(buffer_size=batch_size)
        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.shuffle(1000 + 3 * batch_size)

        dataset = dataset.repeat(num_epochs)
        self.dataset = dataset.batch(batch_size)


    def decode(self, serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                # 'image/filename': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert from a scalar string tensor to a float32 tensor with shape
        image_decoded = tf.image.decode_png(features['image/encoded'], channels=3)
        image = tf.image.resize_image_with_pad(image_decoded,
                                               self.resize_h,
                                               self.resize_w)

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['image/label'], tf.int64)

        return image, label


    def augment(self, image, label):
        """Placeholder for data augmentation.
        """
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.rot90(image, k=random.randint(0,4))
        # image = tf.image.random_brightness(image, max_delta=0.5)
        # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.04)
        # image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        # image = tf.image.resize_images(image, [self.resize_h, self.resize_w])

        return image, label


    def normalize(self, image, label):
        # """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
        # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        return tf.div(tf.subtract(image, MEAN), STD), label
