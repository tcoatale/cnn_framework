import tensorflow as tf

def sparse_cross_entropy(logits, sparse_labels, classes, name):
  logits = tf.cast(logits, tf.float64)
  dense = tf.cast(tf.equal(tf.range(0, classes), tf.cast(sparse_labels, dtype=tf.int32)), dtype=tf.float64)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, dense), name='cross_entropy')
  loss = tf.cast(loss, tf.float32, name=name)
  rect_loss = tf.clip_by_value(loss, 1e-4, 1e4, name= name+'_rect')
  return rect_loss
  
def classification_rate(logits, labels):
  preds = tf.cast(logits, tf.float32)
  targets = tf.reshape(labels, [-1])
  boolean_good_classifications = tf.nn.in_top_k(preds, tf.cast(targets, tf.int32), 1)
  integer_good_classifications = tf.cast(boolean_good_classifications, tf.int32)
  good_classifications = tf.reduce_sum(integer_good_classifications)
  return good_classifications
