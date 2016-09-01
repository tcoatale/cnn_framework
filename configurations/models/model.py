import tensorflow as tf

class Model:
  def __init__(self, model):
    self.architecture_start = model.architecture_start
    self.architecture_end = model.architecture_end
    self.output = model.output
    
  def training_inference(self, image, add_filters, features, training_params, dataset):
    intermediary_output = self.architecture_start(image, add_filters, features)
    dropout_layer_1 = tf.nn.dropout(intermediary_output, training_params.keep_prob[0])
    architecture_output = self.architecture_end(dropout_layer_1, add_filters, features)
    dropout_layer_2 = tf.nn.dropout(architecture_output, training_params.keep_prob[1])
    return self.output(dropout_layer_2, dataset)
      
  def testing_inference(self, image, add_filters, features, training_params, dataset):
    intermediary_output = self.architecture_start(image, add_filters, features)
    architecture_output = self.architecture_end(intermediary_output, add_filters, features)
    return self.output(architecture_output, dataset)
    
  def inference(self, image, add_filters, features, training_params, dataset, testing=False):
    function = self.testing_inference if testing else self.training_inference
    output = function(image, add_filters, features, training_params, dataset)    
    return output
    