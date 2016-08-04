import tensorflow as tf

class Model:
  def __init__(self, model):
    self.architecture = model.architecture    
    self.output = model.output
    
  def training_inference(self, image, add_filters, features, training_params, dataset):
    architecture_output = self.architecture(image, add_filters)
    dropout_layer = tf.nn.dropout(architecture_output, training_params.keep_prob)
    return self.output(dropout_layer, features, dataset)
      
  def testing_inference(self, image, add_filters, features, training_params, dataset):
    architecture_output = self.architecture(image, add_filters)
    return self.output(architecture_output, features, dataset)
    
  def inference(self, image, add_filters, features, training_params, dataset, testing=False):
    function = self.testing_inference if testing else self.training_inference
    output = function(image, add_filters, features, training_params, dataset)
    #output = tf.clip_by_value(output, 1e-8, 1e0, name='rectified_output')
    
    return output
    