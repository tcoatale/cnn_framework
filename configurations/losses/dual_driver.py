import tensorflow as tf
from configurations.losses.loss_functions import classification_rate, sparse_cross_entropy

def training_loss(dataset, logits, labels):
  with tf.variable_scope("training"):
    augmentation_labels, true_labels = dataset.split_labels(labels)
    augmentation_logits, true_logits = logits
    
    true_loss = sparse_cross_entropy(true_logits, true_labels, dataset.classes, name='true_loss')
    augmentation_loss = sparse_cross_entropy(augmentation_logits, augmentation_labels, dataset.sub_classes, name='augmentation_loss')
    
    loss = tf.add(
            tf.mul(true_loss, tf.Variable(8 * 1e-1, dtype=tf.float32, trainable=False)), 
            tf.mul(augmentation_loss, tf.Variable(2 * 1e-1, dtype=tf.float32, trainable=False)),             
            name='loss')

  return loss

def evaluation_loss(dataset, logits, labels):  
  with tf.variable_scope("evaluation"):
    _, sparse_labels = dataset.split_labels(labels)
    loss = sparse_cross_entropy(logits, sparse_labels)
  return loss
  
def classirate(dataset, logits, labels):
  _, true_logits = logits

  with tf.variable_scope("evaluation"):
    _, sparse_labels = dataset.split_labels(labels)
    classirate = classification_rate(true_logits, sparse_labels)
  return classirate
  
  