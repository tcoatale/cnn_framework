import tensorflow as tf
from configurations.models.layers import conv2d, red_block, res_block, average_pool_vector

def architecture(input):  
  conv0 = conv2d([7, 7], 16, input, 'conv0', stride=2)
  pool0 = tf.nn.avg_pool(conv0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')

  res_block1 = res_block(pool0, 'res_block1')
  res_block2 = res_block(res_block1, 'res_block2')
  res_block3 = res_block(res_block2, 'res_block3')
  red_block1 = red_block(res_block3, 'red_block1')
  
  res_block4 = res_block(red_block1, 'res_block4')
  res_block5 = res_block(res_block4, 'res_block5')
  res_block6 = res_block(res_block5, 'res_block6')
  res_block7 = res_block(res_block6, 'res_block7')
  red_block2 = red_block(res_block7, 'red_block2')

  res_block8 = res_block(red_block2, 'res_block8')
  res_block9 = res_block(res_block8, 'res_block9')
  res_block10 = res_block(res_block9, 'res_block10')
  res_block11 = res_block(res_block10, 'res_block11')
  res_block12 = res_block(res_block11, 'res_block12')
  res_block13 = res_block(res_block12, 'res_block13')
  red_block3 = red_block(res_block13, 'red_block3')

  res_block14 = res_block(red_block3, 'res_block14')
  res_block15 = res_block(res_block14, 'res_block15')
  res_block16 = res_block(res_block15, 'res_block16')
  
  return res_block16
  
def output(input, classes):
  return average_pool_vector([1, 1], classes, input, 'output')
       
def training_inference(input, keep_prob, classes):
  architecture_output = architecture(input)
  dropout_layer = tf.nn.dropout(architecture_output, keep_prob)
  return output(dropout_layer, classes)
    
def testing_inference(input, keep_prob, classes):
  architecture_output = architecture(input)
  return output(architecture_output, classes)

def inference(input, keep_prob, dataset, testing=False):
    function = testing_inference if testing else training_inference
    return function(input, keep_prob, dataset.classes)