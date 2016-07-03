from helper import initialize_directories

import datasets.driver_augmented as dataset
import training.config1 as training_params
import models.inception_reset as model

name = 'config1'
log_dir = 'log'
ckpt_dir = 'ckpt'

initialize_directories(log_dir, ckpt_dir, dataset.name, name)

def inference(input, testing=False):
    return None, model.inference(input, training_params.keep_prob, dataset, testing)
    
#%%