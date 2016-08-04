# -*- coding: utf-8 -*-
from configurations.models.alex.alex import architecture as base_architecture
from configurations.models.blocks.output_blocks import fc_output, fc_stream
import tensorflow as tf

#%%
def architecture(image, add_filters):
  augmented_image = tf.concat(3, [image, add_filters])
  return base_architecture(augmented_image)
  
def output(input, features, dataset):
  with tf.variable_scope("feat_augmentation"):
    linear_features = fc_stream(input, [256], name='features')
    all_features = tf.concat(1, [linear_features, features])
  out = fc_output(all_features, dataset.classes, [192, 128], name='output')
  return out
  
