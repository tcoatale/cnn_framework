# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from functools import reduce
from configurations.datasets.pcle.pcle import *

size = 64
original_shape=[64, 64, 1]
data_dir = os.path.join('data', name, 'processed', str(size))

imshape=[64, 64, 1]
additional_filters = 4

n_input = reduce(int.__mul__, imshape)


#%%
def process_inputs(image, add_filter, distort=True):
  augmented_image = tf.concat(2, [image, add_filter])
  width, height, depth = imshape
  aug_shape = [width, height, depth + additional_filters]

  if distort:
    augmented_image = tf.random_crop(augmented_image, aug_shape)
    #augmented_image = tf.image.random_brightness(augmented_image, max_delta=65)
    #augmented_image = tf.image.random_contrast(augmented_image, lower=0.20, upper=1.80)
  else:
    width, height, _ = imshape
    augmented_image = tf.image.resize_image_with_crop_or_pad(augmented_image, width, height)

  simple_image = tf.slice(augmented_image, [0, 0, 0], imshape, name='image_slicer')
  aug_filter = tf.slice(augmented_image, [0, 0, depth], [width, height, additional_filters], name='feature_slicer')

  float_simple_image = tf.image.per_image_whitening(simple_image)  
  float_aug_filter = tf.image.per_image_whitening(aug_filter)  
  
  
  transposed_float_image = tf.transpose(float_simple_image, [1, 0, 2])
  transposed_aug_filter = tf.transpose(float_aug_filter, [1, 0, 2])
  return transposed_float_image, transposed_aug_filter
