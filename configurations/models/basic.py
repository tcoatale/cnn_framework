from configurations.models.blocks.layers import conv2d_layer, pool_layer, normalize
from configurations.models.blocks.output_blocks import fc_output

#%%
def architecture(input):
  print(input.get_shape())
  conv1 = conv2d_layer(input, [5, 5], 64, name='conv1')
  pool2 = pool_layer(normalize(conv1), 2, name='pool2')
  conv3 = conv2d_layer(pool2, [3, 3], 64, name='conv3')
  pool4 = normalize(pool_layer(conv3, 2, name='pool4'))
  print(pool4.get_shape())
  return pool4
  
def output(input, dataset):
  return fc_output(input, dataset, [394, 192])
