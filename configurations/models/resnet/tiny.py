from configurations.models.blocks.architecture_blocks import resnet_starter
from configurations.models.blocks.architecture_blocks import resnet_macro_block as block
from configurations.models.blocks.output_blocks import fc_output, average_pool_vector

#%%
def architecture(input, add_filters):
  print(input.get_shape())
  start = resnet_starter(input, 64)
  macro_b1 = block(start, channels = 128, name='macro_b1')
  macro_b2 = block(macro_b1, channels = 256, name='macro_b2')
  print(macro_b2.get_shape())
  return macro_b2
  
def output(input, features, dataset):
  classes = dataset.classes
  reduction = average_pool_vector([3, 3], classes ** 2, input, name='pool_reduce_out')
  return fc_output(reduction, dataset.classes, [192])
