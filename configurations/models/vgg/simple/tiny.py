import tensorflow as tf
from configurations.models.layers import vgg_block, local_layer
  
#%%
def architecture(input):
  block1 = vgg_block(input, 'vgg_block_1', 32)
  block2 = vgg_block(block1, 'vgg_block_2', 64)
  block3 = vgg_block(block2, 'vgg_block_3', 64)
  block4 = vgg_block(block3, 'vgg_block_4', 96)
  return block4

def output(input, training_params, dataset):
  reshape = tf.reshape(input, [training_params.batch_size, -1])
  fc1 = local_layer(64, reshape, 'fc1')
  fc2 = local_layer(64, fc1, 'fc2')  
  return local_layer(dataset.classes, fc2, 'output')