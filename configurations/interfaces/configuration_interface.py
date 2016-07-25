from configurations.configuration import Configuration
from configurations.interfaces.dataset_interface import get_dataset
from configurations.interfaces.loss_interface import get_loss
from configurations.interfaces.training_interface import get_params
from configurations.interfaces.model_interface import get_model

def get_config(argv):
  if argv and len(argv) == 7:
    dataset_name, dataset_size, training, loss_name, model_name, model_size = argv[1:] 
  else:
    dataset_name = 'driver'
    dataset_size = '64'
    training = 'fast'
    loss_name = 'driver'
    model_name = 'alex'
    model_size = 'small'
    
  dataset = get_dataset(dataset_name, dataset_size)
  loss = get_loss(loss_name)
  training_params = get_params(training)
  model = get_model(model_name, model_size)  

  name = '_'.join([dataset_name, dataset_size, training, loss_name, model_name, model_size])
  freqs = {'display': 10, 'summary': 50, 'save': 500}

  config = Configuration(name, dataset, training_params, loss, model, freqs)
  return config