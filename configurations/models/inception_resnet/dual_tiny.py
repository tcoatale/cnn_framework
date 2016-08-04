# -*- coding: utf-8 -*-
from configurations.models.blocks.architecture_blocks import resnet_starter
from configurations.models.blocks.architecture_blocks import resnet_inception_macro_block as block
from configurations.models.blocks.output_blocks import fc_output, average_pool_vector

import tensorflow as tf

#%%
def architecture(input):
  print(input.get_shape())
  start = resnet_starter(input, 64)
  macro_b1 = block(start, channels = 128, name='macro_b1')
  macro_b2 = block(macro_b1, channels = 256, name='macro_b2')
  print(macro_b2.get_shape())
  return macro_b2
  
def output(input, dataset):
  classes = dataset.classes
  augmentation_classes = dataset.aug_classes
  
  augmentation_reduction = average_pool_vector([3, 3], augmentation_classes ** 2, input, name='pool_reduce_aug')
  augmentation_output =  fc_output(augmentation_reduction, augmentation_classes, [92])
  
  reduction = average_pool_vector([3, 3], augmentation_classes ** 2, input, name='pool_reduce_out')
  reduction_concat = tf.concat(1, [reduction, augmentation_output])
  final_output = fc_output(reduction_concat, classes, [192])
  
  return augmentation_output, final_output