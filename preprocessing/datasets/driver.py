# -*- coding: utf-8 -*-
import os
import glob
from random import shuffle
from functools import reduce

from preprocessing.extractors.gabor_extractor import GaborExtractionManager
from preprocessing.extractors.isomap_extractor import ISOFeatureManager
from preprocessing.extractors.hog_extractor import HogExtractionManager

from preprocessing.image_managers.driver import ImageManager

app_raw_data_root = os.path.join('data', 'raw', 'driver')
data_dir = os.path.join(app_raw_data_root, 'train')
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
  training_files, testing_files = get_split_data_sets(train_ratio, training_files)
    
  return {'train': training_files, 'test': testing_files}

def get_all_files():
  files_by_type = get_files_by_type()
  all_files = reduce(list.__add__, files_by_type.values())
  return all_files
  
def run_extractions():
  gabor_file = 'gabor_features.csv'
  gabor_isomap_file = 'gabor_isomap_features.csv'
  n_components=10
  
  print('Starting Gabor feature extraction')
  gabor_manager = GaborExtractionManager(get_all_files(), gabor_dir, gabor_file)
  gabor_manager.run_extraction()
  
  print('Starting Isomap feature extraction')
  isomap_manager = ISOFeatureManager(gabor_dir, gabor_file, gabor_isomap_file, n_components)
  isomap_manager.run_extraction()
  
  print('Starting Hog feature extraction')
  hog_augmentation_manager = HogExtractionManager(image_files=get_all_files(), 
                                                    dest_dir=hog_dir, 
                                                    pixels_per_cell=12, 
                                                    orientations=8)
  
  hog_augmentation_manager.run_extraction()

