# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from datetime import datetime
import pandas as pd

from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from skimage import color
from skimage import io

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
    
  def format_output(self, output):
    df_dict = {}
    cols = list(range(output.shape[1]))
    
    for col in cols:
      df_dict[col] = output[:, col]
    df_dict['file'] = self.image_files
    df = pd.DataFrame(df_dict)
    return df
    
  def write_to_csv(self, df):
    dest_file = os.path.join(self.dest_dir, self.dest_file)
    df.to_csv(dest_file, index=False)
    
  def run_extraction(self):
    features = np.array(list(map(self.wrap_extraction, enumerate(self.image_files))))
    df = self.format_output(features)
    self.write_to_csv(df)