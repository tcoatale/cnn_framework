from configurations.models.layers import conv2d, local_layer
import tensorflow as tf

def architecture(input):  
  conv0 = conv2d([3, 3], 8, input, 'conv0')
  conv1 = conv2d([3, 3], 8, conv0, 'conv1')
  
  reshape = tf.reshape(conv1, [conv1.get_shape()[0].value, -1])
  fc2 = local_layer(128, reshape, 'fc1')
  return fc2
  
def output(input, training_params, dataset):
  return local_layer(dataset.classes, input, 'output')