from configurations.helper import initialize_directories

import datasets.driver_augmented as dataset
import training.config1 as training_params
import losses.driver_augmented as loss
import models.inception_reset as model

name = 'config1'
log_dir = 'log'
ckpt_dir = 'ckpt'

log_dir, ckpt_dir = initialize_directories([log_dir, ckpt_dir], dataset.name, name)

def inference(input, testing=False):
    return None, model.inference(input, training_params.keep_prob, dataset, testing)
    
def training_loss(logits, labels):
    return loss.training_loss(dataset, logits, labels)
    
def evaluation_loss(logits, labels):
    return loss.training_loss(dataset, logits, labels)
    
#%%
display_freq = 10
eval_freq = 300
summary_freq = 50
save_freq = 1000