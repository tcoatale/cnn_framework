import tensorflow as tf
from configurations.models.layers import conv2d, local_layer
  
#%%
def architecture(input):
  conv1 = conv2d([3, 3], 64, input, 'conv1')
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  
  conv2 = conv2d([1, 1], 64, pool1, 'conv2')
  conv3 = conv2d([3, 3], 64, conv2, 'conv3')
  conv4 = conv2d([1, 1], 64, conv3, 'conv4')
  pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  
  conv5 = conv2d([3, 3], 64, pool2, 'conv5')
  conv6 = conv2d([3, 3], 64, conv5, 'conv6')
  conv7 = conv2d([1, 1], 128, conv6, 'conv7')
  conv8 = conv2d([3, 3], 128, conv7, 'conv8')
  pool3 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
  
  conv9 = conv2d([3, 3], 128, pool3, 'conv9')
  conv10 = conv2d([3, 3], 128, conv9, 'conv10')
  conv11 = conv2d([3, 3], 128, conv10, 'conv11')
  pool4 = tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
  
  conv12 = conv2d([3, 3], 128, pool4, 'conv12')
  conv13 = conv2d([1, 1], 256, conv12, 'conv13')
  conv14 = conv2d([3, 3], 256, conv13, 'conv14')
  conv15 = conv2d([1, 1], 256, conv14, 'conv15')
  pool5 = tf.nn.max_pool(conv15, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
  
  return pool5

def output(input, training_params, dataset):
  reshape = tf.reshape(input, [training_params.batch_size, -1])
  fc1 = local_layer(64, reshape, 'fc1')
  fc2 = local_layer(64, fc1, 'fc2')  
  return local_layer(dataset.classes, fc2, 'output')