# -*- coding: utf-8 -*-
from formatting.batch_manager import BatchManager
from formatting.image_manager import ImageManager
import os

class PreprocessingManager:
  def __init__(self, dataset):    
    self.resize = (dataset.data['images']['resized']['width'], dataset.data['images']['resized']['height'])
    self.batch_size = dataset.data['images_per_batch_file']
    self.files = dataset.get_files_by_type()
    self.image_manager = ImageManager(dataset)
    self.dest_dir = dataset.data['directories']['processed']
    
  def wrap_batch_run(self, type):
    batch_manager = BatchManager(self.files[type],
                                 self.image_manager, 
                                 self.batch_size, 
                                 type,
                                 self.dest_dir)
    
    batch_manager.run()
    
  def run(self):
    if not os.path.isdir(self.dest_dir):
      os.mkdir(self.dest_dir)
    for type in self.files.keys():
      print('Processing', type, 'batches')
      self.wrap_batch_run(type)
      
