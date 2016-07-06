import tensorflow as tf

class Model:
  def __init__(self, model):
    self.architecture = model.architecture    
    self.output = model.output
    
  def training_inference(self, input, training_params, dataset):
    architecture_output = self.architecture(input)
    dropout_layer = tf.nn.dropout(architecture_output, training_params.keep_prob)
    return self.output(dropout_layer, training_params, dataset)
      
  def testing_inference(self, input, training_params, dataset):
    architecture_output = self.architecture(input)
    return self.output(architecture_output, training_params, dataset)
    
  def inference(self, input, training_params, dataset, testing=False):
    function = self.testing_inference if testing else self.training_inference
    return function(input, training_params, dataset)