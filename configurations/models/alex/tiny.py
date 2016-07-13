from configurations.models.layers import conv2d, pool_layer
from configurations.models.blocks import fc_output

#%%
def architecture(input):
  conv1 = conv2d([11, 11], 96, input, name='conv1')
  pool2 = pool_layer(conv1, 2, name='pool2')
  conv3 = conv2d([5, 5], 256, pool2, name='conv3')
  pool4 = pool_layer(conv3, 2, name='pool4')
  conv5 = conv2d([3, 3], 384, pool4, name='conv5')
  conv6 = conv2d([3, 3], 384, conv5, name='conv6')
  conv7 = conv2d([3, 3], 256, conv6, name='conv7')
  pool8 = pool_layer(conv7, 2, name='pool8')
  return pool8
  
def output(input, training_params, dataset):
  return fc_output(input, dataset, [384, 192])