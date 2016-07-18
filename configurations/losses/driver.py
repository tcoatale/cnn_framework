from configurations.losses.loss_functions import cross_entropy
import tensorflow as tf

def training_loss(dataset, logits, labels):
  with tf.variable_scope("training"):    
    _, boolean_dense_labels = dataset.split_labels(labels)
    dense_labels = tf.to_int32(boolean_dense_labels)
    loss = cross_entropy(logits, dense_labels)
  return loss
  
def evaluation_loss(dataset, logits, labels):  
  with tf.variable_scope("evaluation"):
    _, boolean_dense_labels = dataset.split_labels(labels)
    dense_labels = tf.to_int32(boolean_dense_labels)
    loss = cross_entropy(logits, dense_labels)
  return loss