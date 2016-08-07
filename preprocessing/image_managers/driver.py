# -*- coding: utf-8 -*-
import os
import numpy as np
import struct
import skimage.io
import skimage.transform
import pandas as pd

aug_data_root = os.path.join('data', 'augmented', 'driver')
gabor_dir = os.path.join(aug_data_root, 'gabor')
hog_dir = os.path.join(aug_data_root, 'hog')


class ImageManager:
  def __init__(self, resize):
    self.resize = resize
    self.aug_features = pd.read_csv(os.path.join(gabor_dir, 'gabor_isomap_features.csv'))
    
  def load_file(self, file, type):
    file_id = self.get_file_id(file)
    label = self.get_label(file, type)
    image = self.get_image(file)
    aug_filters = self.get_aug_filters(file)
    aug_features = self.get_aug_features(file)

    full_line = np.hstack([file_id, label, image, aug_filters, aug_features])
    return full_line
    
  def get_label(self, file, type):
    if type == 'submission':
      return np.array([0, 0], dtype=np.uint8)
    else:
      dir, file = os.path.split(file)
      dir, file = os.path.split(dir)    
      true_label = int(file[1])
      augmentation = int(true_label in range(1, 4))
      return np.array([true_label, augmentation], dtype=np.uint8)
    
  def get_file_id(self, file):
    dir, file_name = os.path.split(file)
    file_id = int(file_name[4:-4])
    byte_form_file_id = np.array(list(struct.unpack('4B', struct.pack('>I', file_id))), dtype=np.uint8)
    return byte_form_file_id
    
  def get_image(self, file):
    image = skimage.io.imread(file)
    resized_image = skimage.transform.resize(image, self.resize)
    transposed_image = np.transpose(resized_image, [2, 0, 1])
    flattened = np.reshape(transposed_image, [-1])
    int_image = np.array(flattened * 255, np.uint8)
    return int_image

  def get_aug_filters(self, file):
    file_name = os.path.split(file)[-1]
    aug_file = os.path.join(hog_dir, file_name)
    augmentation = skimage.io.imread(aug_file)
    resized_augmentation = skimage.transform.resize(augmentation, tuple(self.resize))
    flattened = np.reshape(resized_augmentation, [-1])
    int_image = np.array(flattened * 255, np.uint8)
    return int_image
    
  def get_aug_features(self, file):
    line = self.aug_features[self.aug_features.file == file]
    values = np.array(line.drop('file', 1).iloc[0].tolist())
    int_values = np.array(255 * values, dtype=np.uint8)
    return int_values

