from configurations.losses.loss_functions import sparse_cross_entropy
import tensorflow as tf

def training_loss(dataset, logits, labels):
  with tf.variable_scope("training"):
    _, sparse_labels = dataset.split_labels(labels)
    loss = sparse_cross_entropy(logits, tf.to_int32(sparse_labels))
  return loss
  
def evaluation_loss(dataset, logits, labels):  
  with tf.variable_scope("evaluation"):
    _, sparse_labels = dataset.split_labels(labels)
    loss = sparse_cross_entropy(logits, tf.to_int32(sparse_labels))
  return loss