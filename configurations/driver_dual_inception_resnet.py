from configurations.configuration import Configuration

import configurations.datasets.driver.t_128 as dataset
import configurations.training.config1 as training_params
import configurations.losses.driver_augmented as loss
import configurations.models.inception_residual.dual.tiny as model

def get_config():
  name = 'dual_inception_resnet'
  freqs = {'display': 10, 'eval': 300, 'summary': 50, 'save': 1000}
  config = Configuration(name, dataset, training_params, loss, model, freqs)
  
  return config