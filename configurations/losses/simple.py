# -*- coding: utf-8 -*-
from configurations.losses.loss_functions import classification_rate, sparse_cross_entropy

def training_loss(dataset, logits, labels):
  loss = sparse_cross_entropy(logits, labels, dataset.classes, name='loss_training')
  return loss
  
def evaluation_loss(dataset, logits, labels):  
  loss = sparse_cross_entropy(logits, labels, dataset.classes, name='loss_eval')
  return loss
  
def classirate(dataset, logits, labels):
  classirate = classification_rate(logits, labels)
  return classirate
