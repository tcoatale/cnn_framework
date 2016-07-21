from configurations.configuration import Configuration

import configurations.datasets.driver.t_32 as dataset
import configurations.training.fast as training_params
import configurations.losses.driver as loss
import configurations.models.alex.small as model

def get_config():
  name = 'normal_alex_32'
  freqs = {'display': 10, 'summary': 50, 'save': 500}
  config = Configuration(name, dataset, training_params, loss, model, freqs)
  return config