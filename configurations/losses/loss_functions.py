import tensorflow as tf

def sparse_cross_entropy(logits, sparse_labels):  
  flat_logits = tf.reshape(logits, [-1])
  ranges = tf.range(tf.shape(logits)[0]) * tf.shape(logits)[1]

  indices = ranges + tf.to_int32(sparse_labels)
  loss = tf.cast(-tf.reduce_mean(tf.log(tf.gather(flat_logits, indices))), tf.float32)
  
  return loss

def cross_entropy(logits, dense_labels):
  sparse_labels = tf.argmax(dense_labels, dimension=1)  
  return sparse_cross_entropy(logits, sparse_labels)

def classification_rate(logits, labels):
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  return top_k_op