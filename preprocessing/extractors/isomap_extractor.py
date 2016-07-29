# -*- coding: utf-8 -*-
import os
from sklearn.manifold import Isomap
import pandas as pd

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
    cols = list(range(reduced_normalized_features.shape[1]))
      
    for col in cols:
      df_dict[col] = reduced_normalized_features[:, col]
    df_dict['file'] = files
    df = pd.DataFrame(df_dict)
    df.to_csv('gabor_features.csv', index=False)
    return df
    
  def write_output(self, output):
    dest_file = os.path.join(self.dir, self.dest_file)
    output.to_csv(dest_file, index=False)

  def run_extraction(self):
    path_to_file = os.path.join(self.dir, self.feature_file)
    original_features = pd.read_csv(path_to_file)

    files = self.get_files(original_features)
    reduced_normalized_features = self.compute_iso_map(original_features)
    output = self.format_output(files, reduced_normalized_features)
    self.write_output(output)