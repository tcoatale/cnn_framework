import tensorflow as tf
from configurations.models.layers import conv2d, red_block, res_block, average_pool_vector

def architecture(input):  
  conv0 = conv2d([7, 7], 64, input, 'conv0', stride=2)
  pool0 = tf.nn.avg_pool(conv0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')

  res_block1 = res_block(pool0, 'res_block1')
  res_block2 = res_block(res_block1, 'res_block2')
  red_block1 = red_block(res_block2, 'red_block1')
  
  res_block4 = res_block(red_block1, 'res_block3')
  res_block5 = res_block(res_block4, 'res_block4')
  red_block2 = red_block(res_block5, 'red_block2')

  res_block8 = res_block(red_block2, 'res_block5')
  red_block3 = red_block(res_block8, 'red_block3')
  
  return red_block3

def output(input, training_params, dataset):
  return average_pool_vector([1, 1], dataset.classes, input, 'output')