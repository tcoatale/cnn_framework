import tensorflow as tf
from configurations.models.blocks.layers import flat, fc_layer, readout_layer, conv2d_layer

def fc_output(input, dataset, fc_units_list):
  reshape = flat(input)
  fcs = [fc_layer(reshape, fc_units_list[0], name='fc1')]
  for i in range(1, len(fc_units_list)):
    fc = fc_layer(fcs[i-1], fc_units_list[i], name='fc' + str(i+1))
    fcs += [fc]
    
  normalized_output = tf.nn.l2_normalize(readout_layer(fcs[-1], dataset.classes, name='normalized_output'), dim=1)
  rectified_output = tf.nn.relu(normalized_output, name='rectified_output')

  return rectified_output
  
  
def average_pool_vector(conv_shape, outputs, input, name):
  with tf.variable_scope(name):
    outputs_layer = conv2d_layer(input, conv_shape, outputs, 'conv')
    width = input.get_shape()[1].value
    height = input.get_shape()[2].value
    pool_layer = tf.nn.avg_pool(outputs_layer, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='VALID', name='pool')
    flattened_output = flat(pool_layer)  
    
  return flattened_output
  

