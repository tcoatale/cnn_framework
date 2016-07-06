from configurations.configuration import Configuration

import configurations.datasets.driver.t_128 as dataset
import configurations.training.config2 as training_params
import configurations.losses.driver as loss
import configurations.models.vgg.simple.normal as model

def get_config():
  name = 'vggnet'
  freqs = {'display': 10, 'eval': 300, 'summary': 50, 'save': 1000}
  config = Configuration(name, dataset, training_params, loss, model, freqs)
  
  return config