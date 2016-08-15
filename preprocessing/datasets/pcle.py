# -*- coding: utf-8 -*-
import glob 
import os
import numpy as np
from functools import reduce

from preprocessing.extractors.gabor_extractor import GaborExtractionManager
from preprocessing.extractors.blob_extractor import BlobExtractionManager
from preprocessing.extractors.hog_extractor import HogExtractionManager
from preprocessing.extractors.frame_extractor import FrameExtractionManager

from preprocessing.image_managers.pcle import ImageManager

videos_dir = os.path.join('data', 'pcle', 'raw', 'videos')
frames_dir = os.path.join('data', 'pcle', 'raw', 'frames')
dest_dir_base = os.path.join('data', 'pcle', 'processed')

aug_dir = os.path.join('data', 'pcle', 'augmented')
blob_dir = os.path.join(aug_dir, 'blob')
gabor_dir = os.path.join(aug_dir, 'gabor')
hog_dir = os.path.join(aug_dir, 'hog')


#%%
def get_files_of_sequence(seq):
  dir, id = os.path.split(seq)
  id = id.split('.')[0]
  dir, label = os.path.split(dir)
  
  seq_path = os.path.join(frames_dir, '*' + '_'.join([label, id]) + '*')
  frame_files = glob.glob(seq_path)
  return frame_files

def get_files_by_type():
  np.random.seed(213)
  videos_path = os.path.join(videos_dir, '*', '*')
  videos = glob.glob(videos_path)
  training_sequences = np.random.choice(videos, int(0.5 * len(videos)), replace=False).tolist()
  testing_sequences = list(set(videos) - set(training_sequences))
    
  training_files = reduce(list.__add__, list(map(get_files_of_sequence, training_sequences)))
  testing_files = reduce(list.__add__, list(map(get_files_of_sequence, testing_sequences)))
  return {'train': training_files, 'test': testing_files}
  
def get_all_files():
  files_by_type = get_files_by_type()
  all_files = reduce(list.__add__, files_by_type.values())
  return all_files


#%%  
def run_extractions():
  gabor_file = 'gabor_features.csv'
  
  print('Starting frame extraction')
  frame_extraction_manager = FrameExtractionManager(videos_dir=videos_dir, frames_dir=frames_dir, downsample=2)
  frame_extraction_manager.run_extraction()
  
  files = get_all_files()

  print('Starting Blob feature extraction')
  blob_extraction_manager = BlobExtractionManager(files, blob_dir)
  blob_extraction_manager.run_extraction()
  
  print('Starting Gabor feature extraction')
  gabor_manager = GaborExtractionManager(files, gabor_dir, gabor_file)
  gabor_manager.run_extraction()

  print('Starting Hog feature extraction')
  hog_manager = HogExtractionManager(files, hog_dir)
  hog_manager.run_extraction()
