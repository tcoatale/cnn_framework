# -*- coding: utf-8 -*-
from configurations.models.alex.alex import architecture as base_architecture
from configurations.models.blocks.output_blocks import fc_output, fc_stream
import tensorflow as tf

def architecture(image, add_filters):
  augmented_image = tf.concat(3, [image, add_filters])
  return base_architecture(augmented_image)
  
def output(input, features, dataset):
  classes = dataset.classes
  augmentation_classes = dataset.aug_classes
  
  with tf.variable_scope("feat_augmentation"):
    linear_features = fc_stream(input, [256])
    all_features = tf.concat(1, [linear_features, features])
  aug_output = fc_output(all_features, augmentation_classes, [192, 128], name='aug_output')
  
  stream = fc_stream(all_features, [192, 128], name='intermediary_stream')
  stream_concat = tf.concat(1, [stream, aug_output])

  final_output = fc_output(stream_concat, classes, [192], name='true_output')  
  return final_output