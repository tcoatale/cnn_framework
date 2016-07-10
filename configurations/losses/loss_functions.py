import tensorflow as tf

def cross_entropy(logits, labels):
  return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits + 1e-15), reduction_indices=[1]), name='cross_entropy')

def classification_rate(logits, labels):
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  return top_k_op