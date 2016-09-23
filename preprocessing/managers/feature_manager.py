# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd

class FeatureReader:
  def __init__(self, feature_file):
    self.features = pd.read_csv(feature_file)

  def read(self, file):
    line = self.features[self.features.file == file]
    values = np.array(line.drop('file', 1).iloc[0].tolist())
    int_values = np.array(255 * values, dtype=np.uint8)
    return int_values

class FeatureManager:
  def __init__(self, data, raw_dir):
    self.data = data
    self.raw_dir = raw_dir
    
    augmentations = data['augmentations']
    feature_augmentations = filter(lambda a: a['usage'] == 'features', augmentations)
    self.feature_readers = list(map(self.init_feature_reader, feature_augmentations))
    
  def init_feature_reader(self, augmentation):
    feature_file = os.path.join(self.raw_dir, augmentation['output'])
    reader = FeatureReader(feature_file)
    return reader
    
  def get_augmentation_features(self, file):
    result = np.hstack(list(map(lambda r: r.read(file), self.feature_readers)))
    return result