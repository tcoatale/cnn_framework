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
  macro_b2 = block(macro_b1, channels = 384, name='macro_b2')
  macro_b3 = block(macro_b2, channels = 512, name='macro_b3')
  macro_b4 = block(macro_b3, channels = 384, name='macro_b4')
  print(macro_b4.get_shape())
  return macro_b4
  
def output(input, features, dataset):
  classes = dataset.classes
  with tf.variable_scope("feat_augmentation"):
    reduction = average_pool_vector([3, 3], classes ** 3, input, name='pool_reduce_out')
    all_features = tf.concat(1, [reduction, features])
  
  return fc_output(all_features, dataset.classes, [384, 192])

