from configurations.models.blocks.output_blocks import fc_output
from configurations.models.blocks.layers import conv2d_layer, pool_layer
import tensorflow as tf

#%%
def architecture(input, add_filters, features):
  print(input.get_shape())
  conv1 = conv2d_layer(input, [11, 11], 96, name='conv1')
  pool2 = pool_layer(conv1, 2, name='pool2')
  conv3 = conv2d_layer(pool2, [5, 5], 128, name='conv3')
  pool4 = pool_layer(conv3, 2, name='pool4')
  conv5 = conv2d_layer(pool4, [3, 3], 256, name='conv5')
  dropout_layer = tf.nn.dropout(conv5, 0.50)
  conv6 = conv2d_layer(dropout_layer, [3, 3], 256, name='conv6')
  conv7 = conv2d_layer(conv6, [3, 3], 128, name='conv7')
  pool8 = pool_layer(conv7, 2, name='pool8')
  print(pool8.get_shape())
  return pool8
  
def output(input, dataset):  
  return fc_output(input, dataset.classes, [192, 128], name='output')