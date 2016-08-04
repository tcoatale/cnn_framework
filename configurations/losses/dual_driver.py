import tensorflow as tf
from configurations.losses.loss_functions import classification_rate, sparse_cross_entropy

def training_loss(dataset, logits, labels):
  aug_logits, true_logits = logits
  aug_labels, true_labels = dataset.split_labels(labels)
  
  aug_loss = sparse_cross_entropy(aug_logits, aug_labels, dataset.aug_classes, name='loss_training_aug')
  tf.add_to_collection('losses', aug_loss)
  true_loss = sparse_cross_entropy(true_logits, true_labels, dataset.classes, name='loss_training')
  return true_loss
  
def evaluation_loss(dataset, logits, labels):  
  _, true_logits = logits
  _, true_labels = dataset.split_labels(labels)
  
  true_loss = sparse_cross_entropy(true_logits, true_labels, dataset.classes, name='loss_evaluation')
  return true_loss
  
def classirate(dataset, logits, labels):
  _, sparse_labels = dataset.split_labels(labels)
  _, true_logits = logits
  classirate = classification_rate(true_logits, sparse_labels)
  return classirate