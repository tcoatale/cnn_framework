# -*- coding: utf-8 -*-
import os
import numpy as np
import struct
import skimage.io
import skimage.transform
import pandas as pd

app_raw_data_root = os.path.join('data', 'raw', 'pn')
label_dict = {'GBM': 0, 'meningioma': 1}

aug_data_root = os.path.join('data', 'pcle', 'augmented')
gabor_dir = os.path.join(aug_data_root, 'gabor')
blob_dir = os.path.join(aug_data_root, 'blob')
brief_dir = os.path.join(aug_data_root, 'brief')


def byte_form(input):
  return np.array(list(struct.unpack('4B', struct.pack('>I', int(input)))), dtype=np.uint8)


class ImageManager:
  def __init__(self, resize):
    self.resize = resize    
    self.aug_features = pd.read_csv(os.path.join(gabor_dir, 'gabor_features.csv'))

  def load_file(self, file, type):
    file_id = self.get_file_id(file)
    label = self.get_label(file, type)
    image = self.get_image(file)
    aug_filters = self.get_aug_filters(file)
    aug_features = self.get_aug_features(file)
    full_line = np.hstack([file_id, label, image, aug_filters, aug_features])
    return full_line
    
  def get_label(self, file, type):
    dir, fname = os.path.split(file)
    fname = fname.split('.')[0]
    fields = fname.split('_')
    label = label_dict[fields[0]]
    return np.array([label], dtype=np.uint8)
    
  def get_file_id(self, file):
    dir, fname = os.path.split(file)
    fname = fname.split('.')[0]
    fields = fname.split('_')
    
    label, seq, id = fields
    label = label_dict[label]
    
    byte_form_id = byte_form(id)
    label = np.array([label], dtype=np.uint8)
    seq = np.array([seq], dtype=np.uint8)
    
    byte_form_file_id = np.hstack([label, seq, byte_form_id])
    return byte_form_file_id
    
  def get_image(self, file):
    image = skimage.io.imread(file)
    resized_image = skimage.transform.resize(image, self.resize)
    transposed_image = np.transpose(resized_image, [1, 0])
    flattened = np.reshape(transposed_image, [-1])
    int_image = np.array(flattened * 255, np.uint8)
    return int_image
  
  def get_aug_channel(self, dir, file):
    file_name = os.path.split(file)[-1]
    aug_file = os.path.join(dir, file_name)
    augmentation = skimage.io.imread(aug_file)
    resized_augmentation = skimage.transform.resize(augmentation, tuple(self.resize))
    return resized_augmentation
    
  def get_aug_filters(self, file):
    brief_channel = self.get_aug_channel(brief_dir, file)
    blob_channel = self.get_aug_channel(blob_dir, file)
    blob_channel = blob_channel.reshape(blob_channel.shape + (1,))
    augmentation_filters = np.concatenate((brief_channel, blob_channel), 2)    
    flattened = np.reshape(augmentation_filters, [-1])
    int_image = np.array(flattened * 255, np.uint8)
    return int_image
    
  def get_aug_features(self, file):
    '''
    line = self.aug_features[self.aug_features.file == file]
    values = np.array(line.drop('file', 1).iloc[0].tolist())
    int_values = np.array(255 * values, dtype=np.uint8)
    '''
    return np.array([]*96)


