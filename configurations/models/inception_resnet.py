import tensorflow as tf
from configurations.models.layers import conv2d, red_block, inception_res_block, average_pool_vector

def architecture(input):
  conv0 = conv2d([7, 7], 24, input, 'conv0', stride=2)
  pool0 = tf.nn.avg_pool(conv0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')

  inception_res_block1 = inception_res_block(pool0, 'inception_res_block1')
  inception_res_block2 = inception_res_block(inception_res_block1, 'inception_res_block2')
  inception_res_block3 = inception_res_block(inception_res_block2, 'inception_res_block3')
  red_block1 = red_block(inception_res_block3, 'red_block1')
  
  inception_res_block4 = inception_res_block(red_block1, 'inception_res_block4')
  inception_res_block5 = inception_res_block(inception_res_block4, 'inception_res_block5')
  inception_res_block6 = inception_res_block(inception_res_block5, 'inception_res_block6')
  inception_res_block7 = inception_res_block(inception_res_block6, 'inception_res_block7')
  red_block2 = red_block(inception_res_block7, 'red_block2')

  inception_res_block8 = inception_res_block(red_block2, 'inception_res_block8')
  inception_res_block9 = inception_res_block(inception_res_block8, 'inception_res_block9')
  inception_res_block10 = inception_res_block(inception_res_block9, 'inception_res_block10')
  inception_res_block11 = inception_res_block(inception_res_block10, 'inception_res_block11')
  inception_res_block12 = inception_res_block(inception_res_block11, 'inception_res_block12')
  inception_res_block13 = inception_res_block(inception_res_block12, 'inception_res_block13')
  red_block3 = red_block(inception_res_block13, 'red_block3')

  inception_res_block14 = inception_res_block(red_block3, 'inception_res_block14')
  inception_res_block15 = inception_res_block(inception_res_block14, 'inception_res_block15')
  inception_res_block16 = inception_res_block(inception_res_block15, 'inception_res_block16')
  
  return inception_res_block16
  
def output(input, classes):
  return average_pool_vector([1, 1], classes, input, 'output')
       
def training_inference(input, keep_prob, classes):
  architecture_output = architecture(input)
  dropout_layer = tf.nn.dropout(architecture_output, keep_prob)
  return output(dropout_layer, classes)
    
def testing_inference(input, keep_prob, classes):
  architecture_output = architecture(input)
  return output(architecture_output, classes)

def inference(input, training_params, dataset, testing=False):
    function = testing_inference if testing else training_inference
    return function(input, training_params.keep_prob, dataset.classes)