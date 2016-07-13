from configurations.configuration import Configuration

import configurations.datasets.driver.t_64 as dataset
import configurations.training.config3 as training_params
import configurations.losses.driver as loss
import configurations.models.vgg.tiny as model

def get_config():
  name = 'tiny_vgg'
  freqs = {'display': 10, 'summary': 50, 'save': 500}
  config = Configuration(name, dataset, training_params, loss, model, freqs)
  return config