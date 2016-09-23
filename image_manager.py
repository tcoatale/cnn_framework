# -*- coding: utf-8 -*-
from datasets.json_interface import JsonInterface
from preprocessing.managers.channel_manager import ChannelManager
from preprocessing.managers.feature_manager import FeatureManager
import skimage.io
import skimage.transform
import numpy as np
import glob
import os

class ImageManager:
  def __init__(self, data, dataset_label_handler=None):
    self.data = data
    self.max_file_length = 50
    self.resize = (data['images']['resized']['width'], data['images']['resized']['height'])
    self.raw_dir = data['directories']['raw']
    
    self.label_handler = dataset_label_handler
    self.channel_manager = ChannelManager(data, self.raw_dir, self.resize)
    self.feature_manager = FeatureManager(data, self.raw_dir)

  def load_file(self, file):
    file_id = self.get_file_id(file)
    label = self.get_label(file)
    image = self.get_image(file)
    aug_channels = self.get_augmentation_channels(file)
    aug_features = self.get_augmentation_features(file)
    full_line = np.hstack([file_id, label, image, aug_channels, aug_features])
    return full_line
    
  def get_file_id(self, file):
    file_id = list(map(ord, file))
    pad_length = self.max_file_length - len(file_id)
    file_id = [ord(' ')] * pad_length + file_id
    return file_id
    
  def get_label(self, file):
    return self.label_handler(file)
    
  def get_image(self, file):
    image = skimage.io.imread(file)
    resized_image = skimage.transform.resize(image, self.resize)
    transposed_image = np.transpose(resized_image, [1, 0])
    flattened = np.reshape(transposed_image, [-1])
    int_image = np.array(flattened * 255, np.uint8)
    return int_image
    
  def get_augmentation_channels(self, file):
    return self.channel_manager.get_augmentation_channels(file)
  
  def get_augmentation_features(self, file):
    return self.feature_manager.get_augmentation_features(file)
    
#%%
interface = JsonInterface('datasets/pcle/metadata.json')
data = interface.parse()
image_manager = ImageManager(data)

#%%
raw_dir = data['directories']['raw']
files = glob.glob(os.path.join(raw_dir, 'frames', '*'))
file = files[0]

#%%
result = image_manager.load_file(file)

