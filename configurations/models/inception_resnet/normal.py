# -*- coding: utf-8 -*-
from configurations.models.blocks.architecture_blocks import resnet_starter
from configurations.models.blocks.architecture_blocks import resnet_inception_macro_block as block
from configurations.models.blocks.output_blocks import fc_output, average_pool_vector

#%%
def architecture(input):
  print(input.get_shape())
  start = resnet_starter(input, 64)
  macro_b1 = block(start, channels = 128, name='macro_b1')
  macro_b2 = block(macro_b1, channels = 256, name='macro_b2')
  macro_b3 = block(macro_b2, channels = 384, name='macro_b3')
  macro_b4 = block(macro_b3, channels = 256, name='macro_b4')
  macro_b5 = block(macro_b4, channels = 128, name='macro_b5')
  print(macro_b5.get_shape())
  return macro_b5
  
def output(input, dataset):
  classes = dataset.classes
  reduction = average_pool_vector([3, 3], classes ** 2, input, name='pool_reduce_out')
  return fc_output(reduction, dataset, [768, 384, 192])

