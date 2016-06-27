import tensorflow as tf

from layers import conv2d, normalize, local_layer, softmax_layer

def residual_inception(input, name):
  channels = input.get_shape()[3].value
  reduction1 = conv2d([1, 1], channels / 2, input, name + '_reduction1')
  conv11 = conv2d([3, 3], channels, reduction1, name + '_conv11')
  conv12 = conv2d([1, 1], channels / 2, conv11, name + '_conv12')
  conv13 = conv2d([3, 3], 2 * channels, conv12, name + '_conv13')
  
  reduction2 = conv2d([1, 1], channels / 2, input, name + '_reduction2')
  conv21 = conv2d([5, 5], channels, reduction2, name + '_conv21')
  conv22 = conv2d([1, 1], channels / 2, conv21, name + 'conv22')
  conv23 = conv2d([3, 3], 2 * channels, conv22, name + '_conv22')
  
  inception_module = tf.concat(3, [conv13, conv23], name + '_inception')
  inception_reduction = conv2d([1, 1], channels, inception_module, name + '_inception_reduction')
  
  residual_normalized = normalize(tf.add(input, inception_reduction, name=name + '_residual'), name=name + '_residual_norm')
  output= tf.nn.relu(residual_normalized, name=name)
  
  return output
  
def reduction(input, name):
  channels = input.get_shape()[3].value
  
  conv0 = conv2d([1, 1], 2 * channels, input, name + '_conv0')
  
  conv_shield_1 = conv2d([1, 1], channels / 2, conv0, name + '_conv_shield_1')
  conv1 = conv2d([3, 3], channels, conv_shield_1, name + '_conv1', stride=2)
  
  conv_shield_2 = conv2d([1, 1], channels / 2, conv0, name + '_conv_shield_2')  
  conv2 = conv2d([5, 5], channels, conv_shield_2, name + '_conv2')
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name + '_pool2')
  
  concat3 = tf.concat(3, [conv1, pool2], name + '_concat3')
  pool_conv0 = tf.nn.avg_pool(conv0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name + '_pool0')  

  residual_reduction = normalize(tf.add(pool_conv0, concat3, name=name + '_residual'), name=name + '_residual_norm')
  output= tf.nn.relu(residual_reduction, name=name)
  return output
  
#%%
def average_pool_output(conv_shape, classes, input, name):
  class_layer = conv2d(conv_shape, classes, input, name + '_conv')

  width = input.get_shape()[1].value
  height = input.get_shape()[2].value
    
  pool_layer = tf.nn.avg_pool(class_layer, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='VALID', name=name + '_pool')  
  reshape = tf.reshape(pool_layer, [-1, classes])
  output = softmax_layer(classes, reshape, name + '_softmax')
  return output

#%%
def alexnet(input, keep_prob, batch_size, classes):
    conv1 = conv2d([11, 11], 96, input, 'conv1', stride=4)
    pool2 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    conv3 = conv2d([5, 5], 256, pool2, 'conv3')
    pool4 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    conv5 = conv2d([3, 3], 384, pool4, 'conv5')
    conv6 = conv2d([3, 3], 384, conv5, 'conv6')
    conv7 = conv2d([3, 3], 256, conv6, 'conv7')
    pool8 = tf.nn.max_pool(conv7, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool8')
  
    dropout_layer = tf.nn.dropout(pool8, keep_prob)
    reshape = tf.reshape(dropout_layer, [batch_size, -1])
    
    local9 = local_layer(2048, reshape, 'local9')
    local10 = local_layer(2048, local9, 'local10')
    softmax_linear = softmax_layer(classes, local10, 'softmax_layer')
    
    return softmax_linear

#%%
def vggnet(input, keep_prob, batch_size, classes):
    conv1 = conv2d([3, 3], 64, input, 'conv1')
    conv2 = conv2d([3, 3], 64, conv1, 'conv2')
    conv3 = conv2d([1, 1], 64, conv2, 'conv3')
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    conv4 = conv2d([3, 3], 64, pool3, 'conv4')
    conv5 = conv2d([3, 3], 64, conv4, 'conv5')
    conv6 = conv2d([1, 1], 64, conv5, 'conv6')
    pool6 = tf.nn.max_pool(conv6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool6')

    conv7 = conv2d([3, 3], 128, pool6, 'conv7')
    conv8 = conv2d([3, 3], 128, conv7, 'conv8')
    conv9 = conv2d([1, 1], 128, conv8, 'conv9')
    pool9 = tf.nn.max_pool(conv9, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool9')

    conv10 = conv2d([3, 3], 128, pool9, 'conv10')
    conv11 = conv2d([3, 3], 128, conv10, 'conv11')
    conv12 = conv2d([3, 3], 128, conv11, 'conv12')
    pool12 = tf.nn.max_pool(conv12, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool12')

    conv13 = conv2d([3, 3], 256, pool12, 'conv13')
    conv14 = conv2d([3, 3], 256, conv13, 'conv14')
    conv15 = conv2d([3, 3], 256, conv14, 'conv15')
    pool15 = tf.nn.max_pool(conv15, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool15')
    
    dropout_layer = tf.nn.dropout(pool15, keep_prob)
    reshape = tf.reshape(dropout_layer, [batch_size, -1])
    
    local9 = local_layer(2048, reshape, 'local9')
    local10 = local_layer(2048, local9, 'local10')
    softmax_linear = softmax_layer(classes, local10, 'softmax_layer')
    
    return softmax_linear
    
#%%
