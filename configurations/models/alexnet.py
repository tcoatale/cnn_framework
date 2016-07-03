import tensorflow as tf

from layers import conv2d, local_layer
  
#%%
def architecture(input):
  conv1 = conv2d([11, 11], 96, input, 'conv1', stride=4)
  pool2 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
  conv3 = conv2d([5, 5], 256, pool2, 'conv3')
  pool4 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
  conv5 = conv2d([3, 3], 384, pool4, 'conv5')
  conv6 = conv2d([3, 3], 384, conv5, 'conv6')
  conv7 = conv2d([3, 3], 256, conv6, 'conv7')
  pool8 = tf.nn.max_pool(conv7, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool8')

  return pool8
  
def output(input, batch_size, classes):
  reshape = tf.reshape(input, [batch_size, -1])
  fc1 = local_layer(2048, reshape, 'fc1')
  fc2 = local_layer(2048, fc1, 'fc2')  
  return local_layer(classes, fc2, 'output')
       
def training_inference(input, keep_prob, batch_size, classes):
  architecture_output = architecture(input)
  dropout_layer = tf.nn.dropout(architecture_output, keep_prob)
  return output(dropout_layer, batch_size, classes)
    
def testing_inference(input, keep_prob, batch_size, classes):
  architecture_output = architecture(input)
  return output(architecture_output, batch_size, classes)

def inference(input, keep_prob, dataset, testing=False):
  function = testing_inference if testing else training_inference
  return function(input, keep_prob, dataset.batch_size, dataset.classes)