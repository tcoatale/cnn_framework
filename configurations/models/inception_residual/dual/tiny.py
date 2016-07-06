import tensorflow as tf
from configurations.models.layers import local_layer, conv2d, inception_res_block, average_pool_vector, red_block

def architecture(input):  
  conv0 = conv2d([7, 7], 64, input, 'conv0', stride=2)
  res_block1 = inception_res_block(conv0, 'res_block1')
  red_block1 = red_block(res_block1, 'red_block1')  
  res_block2 = inception_res_block(red_block1, 'res_block2')
  res_block3 = inception_res_block(res_block2, 'res_block3')  
  res_block4 = inception_res_block(res_block3, 'res_block4')
  red_block2 = red_block(res_block4, 'red_block2')
  
  return red_block2
  
def output(input, training_params, dataset):
  sub_output = average_pool_vector([1, 1], dataset.sub_classes, input, 'sub_output')
  fc_input = average_pool_vector([1, 1], dataset.classes ** 2, input, 'fc_input')
  concat = tf.concat(1, [sub_output, fc_input], name='concat')
  final_output = local_layer(dataset.classes, concat, 'final_output')
  return sub_output, final_output