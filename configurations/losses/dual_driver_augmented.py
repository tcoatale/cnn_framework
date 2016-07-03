from configurations.losses.loss_functions import cross_entropy
import tensorflow as tf

def training_loss(dataset, logits, labels):
  logits1, true_logits = logits
  labels1, true_labels = dataset.split_labels(labels)

  primary_loss = evaluation_loss(true_logits, true_labels)
  sub_loss  = evaluation_loss(logits1, labels1)

  return tf.add(primary_loss, tf.mul(sub_loss, 0.5))

  
def evaluation_loss(logits, labels):
  return cross_entropy(logits, labels)