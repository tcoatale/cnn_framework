# -*- coding: utf-8 -*-
import tensorflow as tf

class Dataset:
  def __init__(self):
    self.data = None
    
  def process_inputs(self, image, add_filter, distort=True):
    depth, height, width = self.data['images']['normal'].values()
    additional_channels = self.data['images']['additional_channels']
    
    augmented_image = tf.concat(2, [image, add_filter])
  
    if distort:
      augmented_image = tf.image.random_brightness(augmented_image, max_delta=65)
      augmented_image = tf.image.random_contrast(augmented_image, lower=0.20, upper=1.80)

  
    simple_image = tf.slice(augmented_image, [0, 0, 0], [width, height, depth], name='image_slicer')
    aug_filter = tf.slice(augmented_image, [0, 0, depth], [width, height, additional_channels], name='feature_slicer')
  
    float_simple_image = tf.image.per_image_whitening(simple_image)  
    float_aug_filter = tf.image.per_image_whitening(aug_filter)  
    
    transposed_float_image = tf.transpose(float_simple_image, [1, 0, 2])
    transposed_aug_filter = tf.transpose(float_aug_filter, [1, 0, 2])
    return transposed_float_image, transposed_aug_filter
