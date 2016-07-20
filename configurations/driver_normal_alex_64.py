from configurations.configuration import Configuration

import configurations.datasets.driver.t_64 as dataset
import configurations.training.config2 as training_params
import configurations.losses.driver as loss
import configurations.models.alex.small as model

def get_config():
  name = 'normal_alex_64'
  freqs = {'display': 10, 'summary': 50, 'save': 500}
  config = Configuration(name, dataset, training_params, loss, model, freqs)
  return config