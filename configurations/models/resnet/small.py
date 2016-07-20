from configurations.models.blocks.layers import conv2d_layer, pool_layer
from configurations.models.blocks.architecture_blocks import resnet_inception_block
from configurations.models.blocks.output_blocks import fc_output

#%%
def architecture(input):
  print(input.get_shape())
  conv0 = conv2d_layer(input, [11, 11], 96, name='conv0')

  block1 = resnet_inception_block(conv0, 'block1')
  block2 = resnet_inception_block(block1, 'block2')
  pool3 = pool_layer(block2, 2, name='pool3')
  
  block4 = resnet_inception_block(pool3, 'block4')
  block5 = resnet_inception_block(block4, 'block5')
  block6 = resnet_inception_block(block5, 'block6')
  block7 = resnet_inception_block(block6, 'block7')
  pool8 = pool_layer(block7, 2, name='pool8')
  
  block9 = resnet_inception_block(pool8, 'block9')
  block10 = resnet_inception_block(block9, 'block10')
  block11 = resnet_inception_block(block10, 'block11')
  block12 = resnet_inception_block(block11, 'block12')
  block13 = resnet_inception_block(block12, 'block13')
  out_conv = pool_layer(block13, 2, name='out_conv')
  
  print(out_conv.get_shape())
  return out_conv
  
def output(input, dataset):
  return fc_output(input, dataset, [1024, 1024])