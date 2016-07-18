from configurations.configuration import Configuration

import configurations.datasets.pn.t_64 as dataset
import configurations.training.config2 as training_params
import configurations.losses.pn as loss
import configurations.models.resnet.tiny as model

def get_config():
  name = 'tiny_resnet'
  freqs = {'display': 10, 'summary': 50, 'save': 500}
  config = Configuration(name, dataset, training_params, loss, model, freqs)
  return config