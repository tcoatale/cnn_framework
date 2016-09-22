# -*- coding: utf-8 -*-
import os
from sklearn.manifold import Isomap
import pandas as pd
import numpy as np

class ISOFeatureManager:
  def __init__(self, input_file, output_file, n_components=50, ratio=0.05):
    self.feature_file = input_file
    self.dest_file = output_file
    self.n_components = n_components
    self.ratio = ratio
        
  def get_files(self, original_features):
    files = original_features.file
    return files
    
  def compute_iso_map(self, original_features):
    feature_matrix = original_features.drop('file', 1).as_matrix()
    feature_matrix = np.nan_to_num(feature_matrix)
    
    dimen_reductor = Isomap(n_components=self.n_components)
    
    full_size = feature_matrix.shape[0]
    train_size = int(self.ratio * full_size)
    
    row_indices = list(range(full_size))
    feature_training_indices = np.random.choice(row_indices, size = train_size)
    training_feature_matrix = feature_matrix[feature_training_indices, :]
    
    dimen_reductor.fit(training_feature_matrix)    
    reduced_features = dimen_reductor.transform(feature_matrix)
    
    reduced_normalized_features = reduced_features - reduced_features.min(axis=0)
    reduced_normalized_features /= reduced_normalized_features.max(axis=0)
    
    return reduced_normalized_features
    
  def format_output(self, files, reduced_normalized_features):
    df_dict = {}
    cols = list(range(reduced_normalized_features.shape[1]))
      
    for col in cols:
      df_dict[col] = reduced_normalized_features[:, col]
    df_dict['file'] = files
    df = pd.DataFrame(df_dict)
    df.to_csv(self.dest_file, index=False)
    return df
    
  def write_output(self, output):
    output.to_csv(self.dest_file, index=False)

  def run_extraction(self):
    original_features = pd.read_csv(self.feature_file)

    files = self.get_files(original_features)
    reduced_normalized_features = self.compute_iso_map(original_features)
    output = self.format_output(files, reduced_normalized_features)
    self.write_output(output)