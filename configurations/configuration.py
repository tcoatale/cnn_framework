from configurations.helper import initialize_directories
from configurations.models.model import Model
import os 

class Configuration:
  def __init__(self, name, dataset, training_params, loss, model, freqs):
    self.dataset = dataset
    self.training_params = training_params
    self.loss = loss
    self.model = Model(model)
    self.name = name
    
    self.initialize_directories()
    self.initialize_frequencies(freqs)

  def initialize_directories(self):
    dirs = (os.path.join('log', 'board'), os.path.join('log', 'ckpt'))
    self.log_dir, self.ckpt_dir = initialize_directories(dirs, self.dataset.name, self.name)
    
  def initialize_frequencies(self, freqs):
    self.display_freq = freqs['display']
    self.eval_freq = freqs['eval']
    self.summary_freq = freqs['summary']
    self.save_freq = freqs['save']
    
    
  def inference(self, input, testing=False):
    return self.model.inference(input, self.training_params, self.dataset, testing)
    
  def training_loss(self, logits, labels):
      return self.loss.training_loss(self.dataset, logits, labels)
      
  def evaluation_loss(self, logits, labels):
      return self.loss.evaluation_loss(self.dataset, logits, labels)