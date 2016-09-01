# -*- coding: utf-8 -*-
import os
from skimage.feature import hog
from skimage import color, exposure
from skimage import io
from datetime import datetime
import pathos.multiprocessing as mp
import numpy as np
  
processes = 4  
  
class HogExtractionManager:
  def __init__(self, image_files, dest_dir, pixels_per_cell=8, orientations=8):
    self.image_files = image_files
    self.dest_dir = dest_dir
    self.pixels_per_cell = pixels_per_cell
    self.orientations = orientations

  def compute_hog(self, file):
    image = color.rgb2gray(io.imread(file))
    _, hog_image = hog(image, 
                       orientations=self.orientations, 
                       pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                       cells_per_block=(1, 1), 
                       visualise=True)
                       
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    
    return hog_image_rescaled
    
  def write_hog(self, file, hog):
    dest_file = os.path.join(self.dest_dir, os.path.split(file)[-1])
    io.imsave(dest_file, hog)
    
  def extract(self, files, index):
    if index % 50 == 0:
      print(datetime.now(), index)
      
    file = files[index]
    hog = self.compute_hog(file)
    self.write_hog(file, hog)
    
  def run_extraction_batch(self, batch_files):
    list(map(lambda index: self.extract(batch_files, index), range(len(batch_files))))
    
  def split_in_batches(self, n_split):
    image_files = self.image_files
    n_images = len(image_files)    
    indices = np.linspace(0, n_images, n_split+1).astype(np.int)
    batch_indices = zip(indices[:-1], indices[1:])
    batches_of_files = [image_files[b[0]:b[1]] for b in batch_indices]
    
    return batches_of_files
    
  def run_extraction(self):
    batches_of_files = self.split_in_batches(processes)
    print('Split files in', len(batches_of_files), 'batches')
    pool = mp.ProcessingPool(processes)
    pool.map(lambda batch_files: self.run_extraction_batch(batch_files), batches_of_files)
