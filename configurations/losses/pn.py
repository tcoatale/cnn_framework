from configurations.losses.loss_functions import cross_entropy
import tensorflow as tf
  
def training_loss(dataset, logits, labels):
  labels1, true_labels = dataset.split_labels(labels)
  with tf.variable_scope("training"):
    dense = tf.cast(tf.equal(tf.range(0, dataset.classes), tf.cast(true_labels, dtype=tf.int32)), dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, dense), name='cross_entropy')
  return loss
  
def evaluation_loss(dataset, logits, labels):
  labels1, true_labels = dataset.split_labels(labels)
  with tf.variable_scope("evaluation"):
    dense = tf.cast(tf.equal(tf.range(0, dataset.classes), tf.cast(true_labels, dtype=tf.int32)), dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, dense), name='cross_entropy')
  return loss