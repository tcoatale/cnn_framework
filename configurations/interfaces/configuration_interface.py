from configurations.configuration import Configuration
from configurations.interfaces.dataset_interface import get_dataset
from configurations.interfaces.loss_interface import get_loss
from configurations.interfaces.training_interface import get_params
from configurations.interfaces.model_interface import get_model

def get_config(argv):
  if argv and len(argv) == 7:
    dataset_name, dataset_size, training, loss_name, model_name, model_size = argv[1:] 
  else:
    dataset_name = 'pn'
    dataset_size = '64'
    training = 'fast'
    loss_name = 'pn'
    model_name = 'resnet'
    model_size = 'small_aug'
    
  dataset = get_dataset(dataset_name, dataset_size)
  loss = get_loss(loss_name)
  training_params = get_params(training)
  model = get_model(model_name, model_size)  

  name = '_'.join([dataset_name, dataset_size, training, loss_name, model_name, model_size])
  freqs = {'display': 50, 'summary': 250, 'save': 2000}

  config = Configuration(name, dataset, training_params, loss, model, freqs)
  return config