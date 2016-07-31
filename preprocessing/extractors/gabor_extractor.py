# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime
import pandas as pd

from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from skimage import color
from skimage import io

import pathos.multiprocessing as mp

processes = 4

class GaborFeatureExtractor:
  thetas = np.arange(0, 1, 0.33) * np.pi
  sigmas = np.arange(1, 10, 3)
  freqs = np.arange(0, 10, 3)

  def __init__(self):
    self.kernels = self.get_kernels()
    
  def get_kernels(self):
    kernels = []
    for theta in GaborFeatureExtractor.thetas:
        for sigma in GaborFeatureExtractor.sigmas:
            for frequency in GaborFeatureExtractor.freqs:
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels += [kernel]
                
    return kernels
    
  def extract_features(self, image):
    feats = np.zeros((len(self.kernels), 2), dtype=np.double)
    for k, kernel in enumerate(self.kernels):
      filtered = ndi.convolve(image, kernel, mode='wrap')
      feats[k, 0] = filtered.mean()
      feats[k, 1] = filtered.var()
    return feats
      
class GaborExtractionManager:
  shrink = (slice(0, None, 4), slice(0, None, 4))

  def __init__(self, image_files, dest_dir, dest_file):
    self.dest_dir = dest_dir
    self.dest_file = dest_file
    self.image_files = image_files
    self.extractor = GaborFeatureExtractor()
    
  def wrap_extraction(self, line):
    index, file = line
    
    if index % 50 == 0:
      print(datetime.now(), index)
        
    image = color.rgb2gray(io.imread(file))
    image = image[GaborExtractionManager.shrink]
    features = self.extractor.extract_features(image)
    features = features.reshape([-1])
    
    return features
    
  def format_output(self, output, files):
    df_dict = {}
    cols = list(range(output.shape[1]))
    
    for col in cols:
      df_dict[col] = output[:, col]
    df_dict['file'] = files
    df = pd.DataFrame(df_dict)
    return df
    
  def write_to_csv(self, df, batch_index):
    dest_file = os.path.join(self.dest_dir, str(batch_index) + '_' + self.dest_file)
    df.to_csv(dest_file, index=False)
    
  def run_extraction_batch(self, batch_index, batch_files):
    features = np.array(list(map(self.wrap_extraction, enumerate(batch_files))))
    df = self.format_output(features, batch_files)
    self.write_to_csv(df, batch_index)
    
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
    pool.map(lambda i: self.run_extraction_batch(i, batches_of_files[i]), range(len(batches_of_files)))
