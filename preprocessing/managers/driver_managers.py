# -*- coding: utf-8 -*-
import skimage.transform
import skimage.io
import numpy as np
import os
import glob
import struct

data_dir = os.path.join('data', 'raw', 'driver', 'train')
submission_dir = os.path.join('data', 'raw', 'driver', 'test')

training_image_files = glob.glob(os.path.join(data_dir, '*', '*'))
submission_image_files = glob.glob(os.path.join(submission_dir, '*'))

dest_dir_base = os.path.join('data', 'processed', 'driver')

class ImageManager:
  def __init__(self, resize):
    self.resize = resize
    self.hog_dir = os.path.join('data', 'augmented', 'driver', 'hog')
    self.data_types = ['train', 'test', 'submission']
    
  def load_file(self, file, type):
    file_id = self.get_file_id(file)
    image = self.get_image(file)
    label = self.get_label(file, type)
    
    full_line = np.hstack([file_id, image, label])
    return full_line
    
  def get_image(self, file):
    file_name = os.path.split(file)[-1]
    aug_file = os.path.join(self.hog_dir, file_name)
    
    image = skimage.io.imread(file)
    resized_image = skimage.transform.resize(image, (self.resize[0], self.resize[1]))
    
    augmentation = skimage.io.imread(aug_file)
    resized_augmentation = skimage.transform.resize(augmentation, (self.resize[0], self.resize[1]))
    
    transposed_image = np.transpose(resized_image, [2, 0, 1]).tolist()
    transposed_image = list(map(lambda i: np.array(i), transposed_image))
    transposed_image.append(resized_augmentation)
    
    augmented_image = np.array(transposed_image, dtype=np.float32)
    integer_augmented_image = np.array(255 * augmented_image, dtype=np.uint8)

    flattened_augmented_image = np.reshape(integer_augmented_image, [-1])
    return flattened_augmented_image
    
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