# -*- coding: utf-8 -*-
import tensorflow as tf
from models.loss_functions import classification_rate, sparse_cross_entropy
from models.params_interface import get_params
from models.architectures.architecture_interface import get_architecture
from datasets.interface import get_dataset
import os

class Model:
  def __init__(self, dataset_name='pcle', architecture_name='alex_full_1', params_name='standard'):
    self.architecture = get_architecture(architecture_name)
    self.dataset = get_dataset(dataset_name)
    self.params = get_params(params_name)
    
    self.log_dir = os.path.join('log', dataset_name, architecture_name, params_name)
    self.ckpt_dir = os.path.join('ckpt', dataset_name, architecture_name, params_name)    
    
  def training_inference(self, image, add_filters, features):
    intermediary_output = self.architecture._start(image, add_filters, features)
    dropout_layer_1 = tf.nn.dropout(intermediary_output, self.params['keep_prob'][0])
    architecture_output = self.architecture._end(dropout_layer_1, add_filters, features)
    dropout_layer_2 = tf.nn.dropout(architecture_output, self.params['keep_prob'][1])
    return self.architecture._output(dropout_layer_2, self.dataset)
      
  def testing_inference(self, image, add_filters, features):
    intermediary_output = self.architecture._start(image, add_filters, features)
    architecture_output = self.architecture._end(intermediary_output, add_filters, features)
    return self.architecture._output(architecture_output, self.dataset)
    
  def inference(self, image, add_filters, features, testing=False):
    function = self.testing_inference if testing else self.training_inference
    output = function(image, add_filters, features)
    return output
    
  def loss(self, logits, labels):
    loss = sparse_cross_entropy(logits, labels, self.dataset.data['classes'], name='loss_training')
    return loss
    
  def classirate(self, logits, labels):
    classirate = classification_rate(logits, labels)
    return classirate
 
