from configurations.configuration import Configuration

import configurations.datasets.driver.t_64 as dataset
import configurations.training.fast as training_params
import configurations.losses.driver as loss
import configurations.models.alex.tiny as model

def get_config():
  name = 'tiny_vgg'
  freqs = {'display': 10, 'summary': 50, 'save': 500}
  config = Configuration(name, dataset, training_params, loss, model, freqs)
  return config