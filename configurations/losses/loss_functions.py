import tensorflow as tf

def cross_entropy(logits, dense_labels):  
  flat_logits = tf.reshape(logits, [-1])
  ranges = tf.range(tf.shape(logits)[0]) * tf.shape(logits)[1]

  sparse_labels = tf.argmax(dense_labels, dimension=1)
  indices = ranges + tf.to_int32(sparse_labels)
  loss = tf.cast(-tf.reduce_mean(tf.log(tf.gather(flat_logits, indices))), tf.float32)
  '''  print(logits.get_shape())
  loss = tf.nn.softmax_cross_entropy_with_logits(logits, dense_labels)'''
  return loss

def classification_rate(logits, labels):
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  return top_k_op