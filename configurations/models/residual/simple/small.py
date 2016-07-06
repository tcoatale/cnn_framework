import tensorflow as tf
from configurations.models.layers import conv2d, red_block, res_block, average_pool_vector

def architecture(input):  
  conv0 = conv2d([7, 7], 32, input, 'conv0', stride=2)
  res_block1 = res_block(conv0, 'res_block1')
  res_block2 = res_block(res_block1, 'res_block2')
  red_block1 = red_block(res_block2, 'red_block1')
  
  res_block3 = res_block(red_block1, 'res_block3')
  res_block4 = res_block(res_block3, 'res_block4')
  red_block2 = red_block(res_block4, 'red_block2')

  res_block5 = res_block(red_block2, 'res_block5')
  res_block6 = res_block(res_block5, 'res_block6')
  red_block3 = red_block(res_block6, 'red_block3')
  
  return red_block3
  
def output(input, training_params, dataset):
  return average_pool_vector([1, 1], dataset.classes, input, 'output')