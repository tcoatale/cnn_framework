import tensorflow as tf
from configurations.losses.loss_functions import classification_rate

def training_loss(dataset, logits, labels):
  with tf.variable_scope("training"):
    dense = tf.cast(tf.equal(tf.range(0, dataset.classes), tf.cast(labels, dtype=tf.int32)), dtype=tf.float64)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, dense), name='cross_entropy')
    loss = tf.cast(loss, tf.float32, name='loss')
  return loss
  
def evaluation_loss(dataset, logits, labels):
  with tf.variable_scope("evaluation"):
    dense = tf.cast(tf.equal(tf.range(0, dataset.classes), tf.cast(labels, dtype=tf.int32)), dtype=tf.float64)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, dense), name='cross_entropy')
    loss = tf.cast(loss, tf.float32, name='loss')
  return loss
  
def classirate(dataset, logits, labels):
  with tf.variable_scope("evaluation"):
    classirate = classification_rate(logits, labels)
  return classirate
  
  