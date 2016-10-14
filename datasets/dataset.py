# -*- coding: utf-8 -*-
import tensorflow as tf

class Dataset:
  def __init__(self):
    self.data = None
    
  def process_inputs(self, image, add_filter, distort=True):
    image_dim = self.data['images']['resized']
    height, width, depth = image_dim['height'], image_dim['width'], image_dim['depth']
    additional_channels = self.get_number('channel')    
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

  def get_number(self, type):
    augmentations = self.data['augmentations']
    type_augmentations = list(filter(lambda a: a['usage'] == type, augmentations))
    type_numbers = list(map(lambda a: a['number'], type_augmentations))
    return sum(type_numbers)