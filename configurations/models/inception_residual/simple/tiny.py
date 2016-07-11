from configurations.models.blocks import resnet_inception_block
from configurations.models.layers import conv2d_layer, normalize, pool_layer, flat, fc_layer, softmax_layer, readout_layer

def architecture(input):  
  conv0 = conv2d_layer(input, [3, 3], 64, 'conv0')
  pool0 = normalize(pool_layer(conv0, 2, name='pool0'))
  block1 = resnet_inception_block(pool0, 'block1')
  pool1 = normalize(pool_layer(block1, 2, name='pool1'))
  block2 = resnet_inception_block(pool1, 'block2')
  return block2
    
def output(input, training_params, dataset):
  reshape = flat(input)
  fc1 = fc_layer(reshape, 384, name='fc1')
  fc2 = fc_layer(fc1, 192, name='fc2')
  return softmax_layer(readout_layer(fc2, dataset.classes, name='out'))