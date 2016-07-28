# -*- coding: utf-8 -*-
from driver import FileManager
from batch_manager import BatchManager
from random import shuffle

class PreprocessingManager:
  def __init__(self, data_dir, submission_dir, dest_dir, train_ratio, batch_size):
    self.data_dir = data_dir
    self.dest_dir = dest_dir
    self.batch_size = batch_size
    self.get_split_data_sets(data_dir, submission_dir, train_ratio)
    
  def get_split_data_sets(self, data_dir, submission_dir, train_ratio):
    file_manager = FileManager(data_dir, submission_dir)
    training_files = file_manager.get_training_files()
    
    shuffle(training_files)    
    train_size = int(len(training_files) * train_ratio)
    
    self.train_files = training_files[:train_size]
    self.valid_files = training_files[train_size:]
    self.testing_files = file_manager.get_testing_files()
    
  def run(self, image_manager):
    batch_manager = BatchManager(self.train_files, image_manager, self.batch_size, 'train', self.dest_dir)
    batch_manager.run()
    
    batch_manager = BatchManager(self.valid_files, image_manager, self.batch_size, 'test', self.dest_dir)
    batch_manager.run()

    batch_manager = BatchManager(self.testing_files, image_manager, self.batch_size, 'submission', self.dest_dir)
    batch_manager.run()
    

  