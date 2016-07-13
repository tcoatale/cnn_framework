import tensorflow as tf

#%%
def weight_variable(shape, stddev, wd, name):
  with tf.device('/cpu:0'):
    initial = tf.truncated_normal(shape, stddev=stddev)
    var = tf.Variable(initial)
  weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss_' + name)
  tf.add_to_collection('losses', weight_decay)
  return var

def bias_variable(shape, constant):
  with tf.device('/cpu:0'):
    initial = tf.constant(constant, shape=shape)
  return tf.Variable(initial) 
    
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')