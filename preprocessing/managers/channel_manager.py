# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import skimage.io
import skimage.transform

class ChannelReader:
  def __init__(self, directory, resize):
    self.directory = directory
    self.resize = resize
    
  def read(self, file):
    return np.array([0] * np.prod(self.resize), dtype=np.uint8)
    
class SingleChannelReader(ChannelReader):
  def read(self, file):
    file_name = os.path.split(file)[-1]
    aug_file = os.path.join(self.directory, file_name)
    augmentation = skimage.io.imread(aug_file)
    resized_augmentation = skimage.transform.resize(augmentation, self.resize, order=0)
    
    return resized_augmentation

class MultipleChannelReader(ChannelReader):
  def read_single(self, aug_file):
    augmentation = skimage.io.imread(aug_file)
    resized_augmentation = skimage.transform.resize(augmentation, self.resize, order=0)
    return resized_augmentation
    
  def resize(self, image):
    return skimage.transform.resize(image, self.resize, order=0)
    
  def read(self, file):
    file_name = os.path.split(file)[-1]
    file_name, _ = file_name.split('.')
    
    aug_files = glob.glob(file_name + '_*')
    augmentations = map(self.read_single, aug_files)
    resized_augmentations = list(map(self.resize, augmentations))
        
    return np.array(resized_augmentations)
  
class ChannelManager:
  def __init__(self, data, raw_dir, resize):
    self.data = data
    self.resize = resize
    self.raw_dir = raw_dir

  def get_channels(self, file, augmentation):
    channel_directory = os.path.join(self.raw_dir, augmentation['output'])
    channel_number = augmentation['number']
    
    if channel_number == 1 or channel_number == 3:
      reader = SingleChannelReader(channel_directory, self.resize)
    else:      
      reader = MultipleChannelReader(channel_directory, self.resize)
      
    augmentation = reader.read(file)
    
    output_dim = (channel_number,) + (self.resize)
    augmentation = np.reshape(augmentation, output_dim)
    
    return augmentation
    
  def get_augmentation_channels(self, file):
    augmentations = self.data['augmentations']
    channel_augmentations = filter(lambda a: a['usage'] == 'channel', augmentations)
    result = np.vstack(list(map(lambda a: self.get_channels(file, a), channel_augmentations)))
    return result.flatten()