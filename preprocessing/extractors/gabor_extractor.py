# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from datetime import datetime
import pandas as pd
import skimage.color
import skimage.io
import skimage.filters
from scipy import ndimage as ndi
import pathos.multiprocessing as mp

processes = 6

class GaborFeatureExtractor:
  thetas = np.arange(0, 1, 0.33) * np.pi
  sigmas = np.arange(1, 10, 3)
  freqs = np.arange(0, 1.1, 0.5)

  def __init__(self):
    self.kernels = self.get_kernels()
    
  def get_kernels(self):
    kernels = []
    for theta in GaborFeatureExtractor.thetas:
        for sigma in GaborFeatureExtractor.sigmas:
            for frequency in GaborFeatureExtractor.freqs:
                kernel = np.real(skimage.filters.gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels += [kernel]
                
    return kernels
    
  def extract_features_sample(self, image):
    feats = np.zeros((len(self.kernels), 2), dtype=np.double)
    for k, kernel in enumerate(self.kernels):
      filtered = ndi.convolve(image, kernel, mode='wrap')
      feats[k, 0] = filtered.mean()
      feats[k, 1] = filtered.var()
    return feats
    
  def extract_features(self, file):
    freq = 84
    image = skimage.color.rgb2gray(skimage.io.imread(file))
    samples = skimage.util.view_as_blocks(image, (freq, freq))
    samples = samples.reshape([-1] + list(samples.shape[2:]))
    features = list(map(self.extract_features_sample, samples))
    return np.array(features)
      
class GaborExtractionManager:
  def __init__(self, frames_dir, dest_file):
    self.dest_file = dest_file    
    self.frames = glob.glob(os.path.join(frames_dir, '*'))

    self.extractor = GaborFeatureExtractor()
    
  def wrap_extraction(self, line):
    index, file = line
    
    if index % 50 == 0:
      print(datetime.now(), index)
    features = self.extractor.extract_features(file)
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
    dest_dir, _ = os.path.split(self.dest_file)
    
    dest_file = os.path.join(dest_dir, str(batch_index) + '_.csv')
    df.to_csv(dest_file, index=False)
    
  def run_extraction_batch(self, batch_index, batch_files):
    features = np.array(list(map(self.wrap_extraction, enumerate(batch_files))))
    df = self.format_output(features, batch_files)
    self.write_to_csv(df, batch_index)
    
  def split_in_batches(self, n_split):
    frames = self.frames
    n_images = len(frames)    
    indices = np.linspace(0, n_images, n_split+1).astype(np.int)
    batch_indices = zip(indices[:-1], indices[1:])
    batches_of_files = [frames[b[0]:b[1]] for b in batch_indices]
    
    return batches_of_files
    
  def clean_up(self):
    dest_dir, _ = os.path.split(self.dest_file)

    gabor_batch_files = glob.glob(os.path.join(dest_dir, '*_.csv'))
    dataframes = list(map(pd.read_csv, gabor_batch_files))
    full_df = pd.concat(dataframes)

    list(map(os.remove, gabor_batch_files))
    full_df.to_csv(self.dest_file, index=False)
    
  def run_extraction(self):
    batches_of_files = self.split_in_batches(processes)
    print('Split files in', len(batches_of_files), 'batches')
    pool = mp.ProcessingPool(processes)
    pool.map(lambda i: self.run_extraction_batch(i, batches_of_files[i]), range(len(batches_of_files)))
    self.clean_up()
