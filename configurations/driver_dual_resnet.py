from configurations.helper import initialize_directories
import configurations.datasets.driver_augmented as dataset
import configurations.training.config1 as training_params
import configurations.losses.dual_driver_augmented as loss
import configurations.models.dual_inception_resnet as model

name = 'dual'
log_dir = 'log'
ckpt_dir = 'ckpt'

log_dir, ckpt_dir = initialize_directories([log_dir, ckpt_dir], dataset.name, name)

def inference(input, testing=False):
    return model.inference(input, training_params.keep_prob, dataset, testing)
    
def training_loss(logits, labels):
    return loss.training_loss(dataset, logits, labels)
    
def evaluation_loss(logits, labels):
    return loss.evaluation_loss(dataset, logits, labels)
    
#%%
display_freq = 10
eval_freq = 300
summary_freq = 50
save_freq = 1000