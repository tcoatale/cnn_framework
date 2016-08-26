# -*- coding: utf-8 -*-
from configurations.models.blocks.output_blocks import fc_output
from configurations.models.blocks.layers import conv2d_layer, pool_layer
from configurations.models.blocks.architecture_blocks import resnet_block

import tensorflow as tf

#%%
def architecture_start(input, add_filters, features):
  print(input.get_shape())
  conv1 = conv2d_layer(input, [11, 11], 96, name='conv1')
  pool2 = pool_layer(conv1, 2, name='pool2')
  inc_res_block_1 = resnet_block(pool2, 'inc_res_block_1')
  inc_res_block_2 = resnet_block(inc_res_block_1, 'inc_res_block_2')
  inc_res_block_3 = resnet_block(inc_res_block_2, 'inc_res_block_3')
  inc_res_block_4 = resnet_block(inc_res_block_3, 'inc_res_block_4')
  pool5 = pool_layer(inc_res_block_4, 2, name='pool5')
  conv6 = conv2d_layer(pool5, [5, 5], 128, name='conv6')
  
  pool_additional_filters = pool_layer(add_filters, 4, name='pool_additional_filters')
  augmented_image = tf.concat(3, [conv6, pool_additional_filters])
  
  inc_res_block_7 = resnet_block(augmented_image, 'inc_res_block_7')
  inc_res_block_8 = resnet_block(inc_res_block_7, 'inc_res_block_8')
  return inc_res_block_8
  
def architecture_end(input, add_filters, features):
  inc_res_block_9 = resnet_block(input, 'inc_res_block_9')
  inc_res_block_10 = resnet_block(inc_res_block_9, 'inc_res_block_10')
  pool11 = pool_layer(inc_res_block_10, 2, name='pool11')
  print(pool11.get_shape())
  return pool11
  
def output(input, dataset):
  out = fc_output(input, dataset.classes, [128], name='output')
  return out

