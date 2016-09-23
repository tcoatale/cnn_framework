# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from functools import reduce
import numpy as np
import glob 
import struct
import skimage.io
import skimage.transform
import pandas as pd

label_dict = {'GBM': 0, 'meningioma': 1}

class Dataset:
  def __init__(self, data):
    self.data = data
  
  def sparse_to_dense_id(self, file_id_tensor):
    identifier_bytes = self.data['bytes']['identifier']
    
    file_id_tensor_int32 = tf.cast(file_id_tensor, dtype=tf.int32)
    
    multiplier_np = 256**(np.linspace(identifier_bytes-1, 0, num=identifier_bytes))
    multiplier = tf.constant(multiplier_np)
    multiplier_int32 = tf.cast(multiplier, dtype=tf.int32)

    file_id = tf.reduce_sum(tf.mul(file_id_tensor_int32, multiplier_int32))
    return file_id
  
  def retrieve_file_id(self, file_number):
    return 'img_' + str(file_number) + '.jpg'
  
  def process_inputs(self, image, add_filter, distort=True):
    depth, height, width = self.data['images']['normal'].values()
    additional_channels = self.data['images']['additional_channels']
    
    augmented_image = tf.concat(2, [image, add_filter])
  
    if distort:
      augmented_image = tf.image.random_brightness(augmented_image, max_delta=65)
      augmented_image = tf.image.random_contrast(augmented_image, lower=0.20, upper=1.80)

  
    simple_image = tf.slice(augmented_image, [0, 0, 0], [width, height, depth], name='image_slicer')
    aug_filter = tf.slice(augmented_image, [0, 0, depth], [width, height, additional_channels], name='feature_slicer')
  
    float_simple_image = tf.image.per_image_whitening(simple_image)  
    float_aug_filter = tf.image.per_image_whitening(aug_filter)  
    
    transposed_float_image = tf.transpose(float_simple_image, [1, 0, 2])
    transposed_aug_filter = tf.transpose(float_aug_filter, [1, 0, 2])
    return transposed_float_image, transposed_aug_filter
  
  def get_files_of_sequence(self, seq):
    frames_dir = self.data['directories']['frames']
    dir, id = os.path.split(seq)
    id = id.split('.')[0]
    dir, label = os.path.split(dir)  
    seq_path = os.path.join(frames_dir, '*' + '_'.join([label, id]) + '_*')
  
    frame_files = glob.glob(seq_path)
    return frame_files
  
  def get_class_videos(self, label):
    videos_dir = self.data['directories']['videos'] 
    return glob.glob(os.path.join(videos_dir, label, '*'))
    
  def split_class_videos(videos):
    training_sequences = np.random.choice(videos, int(0.8 * len(videos)), replace=False).tolist()
    testing_sequences = list(set(videos) - set(training_sequences))
    
    return [training_sequences, testing_sequences]
      
  def split_videos(self):
    videos_dir = self.data['directories']['videos']
    
    classes = os.listdir(videos_dir)
    classes_videos = list(map(self.get_class_videos, classes))
    splits = list(map(self.split_class_videos, classes_videos))
    
    training_sequences = reduce(list.__add__, (map(lambda x: x[0], splits)))
    testing_sequences = reduce(list.__add__, map(lambda x: x[1], splits))
    
    return training_sequences, testing_sequences
  
  def get_files_by_type(self):
    np.random.seed(213)  
    training_sequences, testing_sequences = self.split_videos()    
    training_files = reduce(list.__add__, list(map(self.get_files_of_sequence, training_sequences)))
    testing_files = reduce(list.__add__, list(map(self.get_files_of_sequence, testing_sequences)))
    
    print('Training size:', len(training_files), 'Testing size:', len(testing_files))
    return {'train': training_files, 'test': testing_files}
    
  def get_all_files(self):
    files_by_type = self.get_files_by_type()
    all_files = reduce(list.__add__, files_by_type.values())
    return all_files
  
  def byte_form(input):
    return np.array(list(struct.unpack('4B', struct.pack('>I', int(input)))), dtype=np.uint8)
  
  
  class ImageManager:
    def __init__(self, resize):
      self.resize = resize    
      self.aug_features = pd.read_csv(os.path.join(gabor_dir, 'gabor_iso_features.csv'))
  
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
      
    def get_brief(self, dir, file):
      file_name = os.path.split(file)[-1]
      aug_file = os.path.join(dir, file_name)
      augmentation = skimage.io.imread(aug_file)
      resized_augmentation = skimage.transform.resize(augmentation, tuple(self.resize), order=0)
      return resized_augmentation
    
    def get_aug_channel(self, dir, file):
      file_name = os.path.split(file)[-1]
      aug_file = os.path.join(dir, file_name)
      augmentation = skimage.io.imread(aug_file)
      resized_augmentation = skimage.transform.resize(augmentation, tuple(self.resize))
      return resized_augmentation
  
      
    def get_aug_filters(self, file):
      brief_channel = self.get_brief(brief_dir, file)
      blob_channel = self.get_aug_channel(blob_dir, file)
      blob_channel = blob_channel.reshape(blob_channel.shape + (1,))
      augmentation_filters = np.concatenate((brief_channel, blob_channel), 2)    
      flattened = np.reshape(augmentation_filters, [-1])
      int_image = np.array(flattened * 255, np.uint8)
      return int_image
      
    def get_aug_features(self, file):
      line = self.aug_features[self.aug_features.file == file]
      values = np.array(line.drop('file', 1).iloc[0].tolist())
      int_values = np.array(255 * values, dtype=np.uint8)
      return int_values
  
  
