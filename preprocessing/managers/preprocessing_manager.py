# -*- coding: utf-8 -*-
from preprocessing.managers.batch_manager import BatchManager
import os

class PreprocessingManager:
  def __init__(self, dataset, resize, batch_size):
    self.dataset = dataset
    self.resize = resize
    self.batch_size = batch_size
    self.files = self.dataset.get_files_by_type()
    
  def run(self):
    dest_dir = os.path.join(self.dataset.dest_dir_base, str(self.resize))
    
    if not os.path.isdir(dest_dir):
      os.mkdir(dest_dir)
      
    image_manager = self.dataset.ImageManager([self.resize, self.resize])
    
    for type in self.files.keys():
      batch_manager = BatchManager(self.files[type],
                                   image_manager, 
                                   self.batch_size, 
                                   type,
                                   dest_dir)
      batch_manager.run()
