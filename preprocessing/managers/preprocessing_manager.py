# -*- coding: utf-8 -*-
from random import shuffle
from preprocessing.managers.batch_manager import BatchManager
import os

class PreprocessingManager:
  def __init__(self, dataset_manager, resize, train_ratio, batch_size):
    self.dataset_manager = dataset_manager
    self.resize = resize
    self.batch_size = batch_size
    self.get_split_data_sets(train_ratio)
    
  def get_split_data_sets(self, train_ratio):
    training_files = self.dataset_manager.training_image_files
    shuffle(training_files)    
    train_size = int(len(training_files) * train_ratio)
    
    self.files = {'train':training_files[:train_size],
                  'test':training_files[train_size:],
                  'submission':self.dataset_manager.submission_image_files}
    
  def run(self):
    dest_dir = os.path.join(self.dataset_manager.dest_dir_base, str(self.resize))
    
    if not os.path.isdir(dest_dir):
      os.mkdir(dest_dir)
      
    image_manager = self.dataset_manager.ImageManager([self.resize, self.resize])
    data_types = image_manager.data_types
    
    for type in data_types:
      batch_manager = BatchManager(self.files[type], 
                                   image_manager, 
                                   self.batch_size, 
                                   type,
                                   dest_dir)
      batch_manager.run()
