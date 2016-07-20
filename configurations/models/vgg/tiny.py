from configurations.models.blocks.layers import conv2d_layer, pool_layer, normalize
from configurations.models.blocks.output_blocks import fc_output


def architecture(input):  
  conv1 = conv2d_layer(input, [5, 5], 64, name='conv1')
  pool2 = normalize(pool_layer(conv1, 2, name='pool2'))
  conv3 = conv2d_layer(pool2, [3, 3], 128, name='conv3')
  pool4 = normalize(pool_layer(conv3, 2, name='pool4'))
  conv5 = conv2d_layer(pool4, [3, 3], 192, name='conv5')
  pool6 = pool_layer(normalize(conv5), 2, name='pool6')
  return pool6
  
def output(input, dataset):
  return fc_output(input, dataset, [384, 192])