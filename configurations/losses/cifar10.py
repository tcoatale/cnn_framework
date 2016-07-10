import tensorflow as tf

def training_loss(dataset, logits, labels):
  with tf.variable_scope("training"):
    dense = tf.cast(tf.equal(tf.range(0, 10), tf.cast(labels, dtype=tf.int32)), dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, dense), name='cross_entropy')
  return loss
  
def evaluation_loss(dataset, logits, labels):
  with tf.variable_scope("evaluation"):
    dense = tf.cast(tf.equal(tf.range(0, 10), tf.cast(labels, dtype=tf.int32)), dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, dense), name='cross_entropy')
  return loss