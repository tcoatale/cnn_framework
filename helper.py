import tensorflow as tf

from layers import conv2d, local_layer, softmax_layer, res_block, red_block, inception_res_block, average_pool_output, average_pool_vector
  
#%%
def inception_resnet(input, keep_prob, classes):
  conv0 = conv2d([7, 7], 16, input, 'conv0', stride=2)
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
  #inception_res_block11 = inception_res_block(inception_res_block10, 'inception_res_block11')
  #inception_res_block12 = inception_res_block(inception_res_block11, 'inception_res_block12')
  #inception_res_block13 = inception_res_block(inception_res_block12, 'inception_res_block13')
  red_block3 = red_block(inception_res_block10, 'red_block3')

  inception_res_block14 = inception_res_block(red_block3, 'inception_res_block14')
  inception_res_block15 = inception_res_block(inception_res_block14, 'inception_res_block15')
  inception_res_block16 = inception_res_block(inception_res_block15, 'inception_res_block16')
  
  dropout_layer = tf.nn.dropout(inception_res_block16, keep_prob)
  average_pool1 = average_pool_vector([1, 1], classes ** 2, dropout_layer, 'average_pool1')
  
  return average_pool1

#%%
def resnet(input, keep_prob, classes):
  conv0 = conv2d([7, 7], 64, input, 'conv0')
  red_block0 = red_block(conv0, 'red_block0')

  res_block1 = res_block(red_block0, 'res_block1')
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
  
  dropout_layer = tf.nn.dropout(res_block16, keep_prob)  
  average_pool1 = average_pool_vector([1, 1], classes ** 2, dropout_layer, 'average_pool1')
  
  return average_pool1

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
  
  return local10

#%%
def vggnet(input, keep_prob, batch_size, classes):
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
  
  dropout_layer = tf.nn.dropout(pool5, keep_prob)
  reshape = tf.reshape(dropout_layer, [batch_size, -1])
  
  local9 = local_layer(1024, reshape, 'local9')
  local10 = local_layer(1024, local9, 'local10')
  
  return local10

#%%
