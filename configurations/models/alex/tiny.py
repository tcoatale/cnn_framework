from configurations.models.layers import conv2d_layer, pool_layer
from configurations.models.blocks import fc_output, resnet_inception_block

#%%
def architecture(input):
  print(input.get_shape())
  conv0 = conv2d_layer(input, [11, 11], 96, name='conv0')
  block1 = resnet_inception_block(conv0, 'block1')
  block2 = resnet_inception_block(block1, 'block2')
  pool3 = pool_layer(block2, 2, name='pool3')
  block4 = resnet_inception_block(pool3, 'block4')
  pool5 = pool_layer(block4, 2, name='pool5')
  block6 = resnet_inception_block(pool5, 'block6')
  block7 = resnet_inception_block(block6, 'block7')
  pool8 = pool_layer(block7, 2, name='pool2')
  return pool8
  
def output(input, dataset):
  return fc_output(input, dataset, [384, 192])