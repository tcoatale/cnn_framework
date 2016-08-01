# -*- coding: utf-8 -*-
import os
import glob
import skimage.transform
import skimage.io
import numpy as np
import struct
from random import shuffle
from functools import reduce
import pandas as pd

app_raw_data_root = os.path.join('data', 'raw', 'driver')
data_dir = os.path.join(app_raw_data_root, 'train')
submission_dir = os.path.join(app_raw_data_root, 'test')
train_ratio = 0.7

aug_data_root = os.path.join('data', 'augmented', 'driver')
gabor_dir = os.path.join(aug_data_root, 'gabor')
hog_dir = os.path.join(aug_data_root, 'hog')

dest_dir_base = os.path.join('data', 'processed', 'driver')


def get_split_data_sets(train_ratio, files):
  shuffle(files)    
  train_size = int(len(files) * train_ratio)  
  return files[:train_size], files[train_size:]
                
def get_files_by_type():
  training_files = glob.glob(os.path.join(data_dir, '*', '*'))
  submission_files = glob.glob(os.path.join(submission_dir, '*'))
  
  training_files, testing_files = get_split_data_sets(train_ratio, training_files)
    
  return {'train': training_files, 'test': testing_files, 'submission':submission_files}

def get_all_files():
  files_by_type = get_files_by_type()
  all_files = reduce(list.__add__, files_by_type.values())
  return all_files

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
    resized_image = skimage.transform.resize(image, (self.resize[0], self.resize[1]))
    transposed_image = np.transpose(resized_image, [2, 0, 1])
    return transposed_image

  def get_aug_filters(self, file):
    file_name = os.path.split(file)[-1]
    aug_file = os.path.join(hog_dir, file_name)
    augmentation = skimage.io.imread(aug_file)
    resized_augmentation = skimage.transform.resize(augmentation, tuple(self.resize))
    return resized_augmentation
    
  def get_aug_features(self, file):
    line = self.aug_features[self.aug_features.file == file]
    return 0
    
