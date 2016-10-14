# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import trange
import time

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
    image_lines = self.load_chunk(i)
    full_data = np.array(image_lines, dtype=np.uint8).reshape(-1)
    binary_data = bytearray(full_data)
    
    batch_name = '_'.join([self.type, 'batch', str(i)])
    batch_file = os.path.join(self.dest_dir, batch_name)
    
    with open(batch_file, "wb") as f:
        f.write(binary_data)
                
  def run(self):
    n_batches = len(self.chunks)
    time.sleep(1)
    
    for batch in trange(n_batches):
      self.write_chunk(batch)