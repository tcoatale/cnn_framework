import tensorflow as tf
  
def training_loss(dataset, logits, labels):
  _, true_labels = dataset.split_labels(labels)
  with tf.variable_scope("training"):
    loss = tf.reduce_mean(tf.abs(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, true_labels)))
  return loss
  
def evaluation_loss(dataset, logits, labels):
  _, true_labels = dataset.split_labels(labels)
  with tf.variable_scope("evaluation"):
    loss = tf.reduce_mean(tf.abs(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, true_labels)))
  return loss
