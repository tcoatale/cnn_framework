# -*- coding: utf-8 -*-
from configurations.models.blocks.architecture_blocks import resnet_starter
from configurations.models.blocks.architecture_blocks import resnet_macro_block as block
from configurations.models.blocks.output_blocks import fc_output, average_pool_vector
import tensorflow as tf


#%%
def architecture(image, add_filters):
  augmented_image = tf.concat(3, [image, add_filters])
  print(augmented_image.get_shape())
  start = resnet_starter(augmented_image, 64)
  macro_b1 = block(start, channels = 128, name='macro_b1')
  macro_b2 = block(macro_b1, channels = 256, name='macro_b2')
  print(macro_b2.get_shape())
  return macro_b2
  
def output(input, features, dataset):
  classes = dataset.classes
  augmentation_classes = dataset.aug_classes

  with tf.variable_scope("feat_augmentation"):
    reduction = average_pool_vector([3, 3], classes ** 3, input, name='pool_reduce_out')
    all_features = tf.concat(1, [reduction, features])
    
  aug_output =  fc_output(all_features, augmentation_classes, [92], name='aug_output')
  
  reduction = average_pool_vector([3, 3], augmentation_classes ** 2, input, name='pool_reduce_out')
  reduction_concat = tf.concat(1, [reduction, aug_output])
  
  final_output = fc_output(reduction_concat, classes, [192], name='true_output')
  
  return aug_output, final_output
