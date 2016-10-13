# -*- coding: utf-8 -*-
import tensorflow as tf
from models.loss_functions import classification_rate, sparse_cross_entropy
from models.training_params.params_interface import ParamsInterface

class Model:
  def __init__(self, architecture, dataset, params_name='standard'):
    self._start = architecture._start
    self._end = architecture._end
    self._output = architecture._output
    self._dataset = dataset
    self.params = ParamsInterface(params_name)
    
  def training_inference(self, image, add_filters, features, training_params):
    intermediary_output = self._start(image, add_filters, features)
    dropout_layer_1 = tf.nn.dropout(intermediary_output, training_params.keep_prob[0])
    architecture_output = self._end(dropout_layer_1, add_filters, features)
    dropout_layer_2 = tf.nn.dropout(architecture_output, training_params.keep_prob[1])
    return self._output(dropout_layer_2, self.dataset)
      
  def testing_inference(self, image, add_filters, features, training_params):
    intermediary_output = self._start(image, add_filters, features)
    architecture_output = self._end(intermediary_output, add_filters, features)
    return self._output(architecture_output, self.dataset)
    
  def inference(self, image, add_filters, features, training_params, testing=False):
    function = self.testing_inference if testing else self.training_inference
    output = function(image, add_filters, features, training_params)
    return output
    
  def loss(self, logits, labels):
    loss = sparse_cross_entropy(logits, labels, self.dataset.classes, name='loss_training')
    return loss
    
  def classirate(self, logits, labels):
    classirate = classification_rate(logits, labels)
    return classirate
 