import tensorflow as tf
from configurations.models.layers import conv2d
from configurations.models.blocks import fc_output

#%%
def architecture(input):
  conv1 = conv2d([11, 11], 96, input, 'conv1')
  pool2 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
  conv3 = conv2d([5, 5], 256, pool2, 'conv3')
  pool4 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
  conv5 = conv2d([3, 3], 384, pool4, 'conv5')
  conv6 = conv2d([3, 3], 384, conv5, 'conv6')
  conv7 = conv2d([3, 3], 256, conv6, 'conv7')
  pool8 = tf.nn.max_pool(conv7, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool8')
  return pool8
  
def output(input, training_params, dataset):
  return fc_output(input, dataset, [384, 192])