import tensorflow as tf

def sparse_cross_entropy(logits, sparse_labels):  
  likelyhood = logits + tf.Variable(1e-5, dtype=tf.float64, trainable=False)
  row_sums = tf.reshape(tf.reduce_sum(likelyhood, 1), [-1, 1])
  probabilities = tf.div(likelyhood, row_sums)
  flat_probabilities = tf.reshape(probabilities, [-1])
 
  ranges = tf.range(tf.shape(logits)[0]) * tf.shape(logits)[1]
  indices = ranges + tf.to_int32(sparse_labels)
  
  cross_entropy = tf.log(tf.gather(flat_probabilities, indices))
  loss = tf.cast(-tf.reduce_mean(cross_entropy), tf.float32, name='loss')
  return loss

def cross_entropy(logits, dense_labels):
  sparse_labels = tf.argmax(dense_labels, dimension=1)  
  return sparse_cross_entropy(logits, sparse_labels)

def classification_rate(logits, labels):
  preds = tf.cast(logits, tf.float32)
  targets = tf.reshape(labels, [-1])
  classirate = tf.reduce_sum(tf.cast(tf.nn.in_top_k(preds, targets, 1), tf.int32)) / targets.get_shape()[0]
  classirate = tf.mul(classirate, tf.Variable(1e2, dtype=tf.float64, trainable=False))
  return classirate