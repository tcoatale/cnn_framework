import tensorflow as tf
from configurations.losses.loss_functions import classification_rate, sparse_cross_entropy

def training_loss(dataset, logits, labels):
  with tf.variable_scope("training"):
    _, sparse_labels = dataset.split_labels(labels)
    loss = sparse_cross_entropy(logits, sparse_labels, dataset.classes, name='loss')
  return loss
  
def evaluation_loss(dataset, logits, labels):  
  with tf.variable_scope("evaluation"):
    _, sparse_labels = dataset.split_labels(labels)
    loss = sparse_cross_entropy(logits, sparse_labels, dataset.classes, name='loss')
  return loss
  
def classirate(dataset, logits, labels):
  with tf.variable_scope("evaluation"):
    _, sparse_labels = dataset.split_labels(labels)
    classirate = classification_rate(logits, sparse_labels)
  return classirate
  
  