import tensorflow as tf

def cross_entropy(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)  
  cross_entropy_mean = tf.reduce_mean(cross_entropy)
  return cross_entropy_mean
  
def classification_rate(logits, labels):
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  return top_k_op