from configurations.configuration import Configuration

import configurations.datasets.pn.t_64 as dataset
import configurations.training.slow as training_params
import configurations.losses.pn as loss
import configurations.models.resnet.small as model

def get_config():
  name = 'small_resnet_64'
  freqs = {'display': 10, 'summary': 50, 'save': 500}
  config = Configuration(name, dataset, training_params, loss, model, freqs)
  return config