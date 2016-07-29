# -*- coding: utf-8 -*-
import os
import glob
import numpy as np

from sklearn.manifold import Isomap

from skimage.filters import gabor_kernel
from skimage import color
from skimage import io

from scipy import ndimage as ndi

from datetime import datetime
import pandas as pd

#%%
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
      
#%%
class GaborExtractionManager:
  shrink = (slice(0, None, 4), slice(0, None, 4))

  def __init__(self, data_dir, dest_dir, dest_file):
    self.data_dir = data_dir
    self.dest_dir = dest_dir
    self.dest_file = dest_file
    self.image_files = glob.glob(os.path.join(data_dir, '*', '*'))
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


class ISOFeatureManager:
  def __init__(self, dir, feature_file, dest_file, n_components):
    self.dir = dir
    self.feature_file = feature_file
    self.dest_file = dest_file
    self.n_components = n_components
        
  def get_files(self, original_features):
    files = original_features.file
    return files
    
  def compute_iso_map(self, original_features):
    feature_matrix = original_features.drop('file', 1).as_matrix()
    
    dimen_reductor = Isomap(n_components=self.n_components)
    reduced_features = dimen_reductor.fit_transform(feature_matrix)
    
    reduced_normalized_features = reduced_features - reduced_features.min()
    reduced_normalized_features /= reduced_normalized_features.max()
    
    return reduced_normalized_features
    
  def format_output(self, files, reduced_normalized_features):
    df_dict = {}
    cols = list(range(self.reduced_normalized_features.shape[1]))
      
    for col in cols:
      df_dict[col] = self.reduced_normalized_features[:, col]
    df_dict['file'] = files
    df = pd.DataFrame(df_dict)
    df.to_csv('gabor_features.csv', index=False)
    return df
    
  def write_output(self, output):
    dest_file = os.path.join(self.dir, self.dest_file)
    output.to_csv(dest_file, index=False)

  def run(self):
    path_to_file = os.path.join(self.dir, self.feature_file)
    original_features = pd.read_csv(path_to_file)

    files = self.get_files(original_features)
    reduced_normalized_features = self.compute_iso_map(original_features)
    output = self.format_output(files, reduced_normalized_features)
    self.write_output(output)

#%%
data_dir = os.path.join('..', 'raw', 'driver', 'train')
dest_dir = os.path.join('..', 'augmented', 'driver', 'gabor')

dest_file = 'gabor_features.csv'

gabor_manager = GaborExtractionManager(data_dir, dest_dir, dest_file)
gabor_manager.run_extraction()