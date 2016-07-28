# -*- coding: utf-8 -*-
import os
import numpy as np

class BatchManager:
  def __init__(self, files, image_manager, batch_size, type, dest_dir):
    self.files = files
    self.image_manager = image_manager
    self.batch_size = batch_size
    self.type = type
    self.dest_dir = dest_dir
    self.split_files()
    
  def split_files(self):
    n_images = len(self.files)
    self.chunks = [self.files[i:i + self.batch_size] for i in range(0, n_images, self.batch_size)]
    
  def load_chunk(self, i):
    files = self.chunks[i]
    image_lines = list(map(lambda file: self.image_manager.load_file(file), files))
    return image_lines
    
  def write_chunk(self, i):
    print('Processing batch number', i, 'of type', self.type, end='\t... ')
    
    image_lines = self.load_chunk(i)
    binary_data = bytearray(np.array(image_lines, dtype=np.uint8).reshape(-1))
    
    batch_name = '_'.join([self.type, 'batch', str(i)])
    batch_file = os.path.join(self.dest_dir, batch_name)

    print('Writing chunk to file', batch_file, end='\t... ')
    
    with open(batch_file, "wb") as f:
        f.write(binary_data)
        
    print('Done.')
        
  def run(self):
    n_batches = len(self.chunks)
    list(map(self.write_chunk, range(n_batches)))