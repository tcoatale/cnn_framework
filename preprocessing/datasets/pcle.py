# -*- coding: utf-8 -*-
import glob 
import os
import numpy as np
from functools import reduce

from preprocessing.extractors.gabor_extractor import GaborExtractionManager
from preprocessing.extractors.blob_extractor import BlobExtractionManager
#from preprocessing.extractors.hog_extractor import HogExtractionManager
from preprocessing.extractors.frame_extractor import FrameExtractionManager
from preprocessing.extractors.brief_extractor import BriefExtractionManager
from preprocessing.extractors.isomap_extractor import ISOFeatureManager
from preprocessing.image_managers.pcle import ImageManager

videos_dir = os.path.join('data', 'pcle', 'raw', 'videos')
frames_dir = os.path.join('data', 'pcle', 'raw', 'frames')
dest_dir_base = os.path.join('data', 'pcle', 'processed')

aug_dir = os.path.join('data', 'pcle', 'augmented')
blob_dir = os.path.join(aug_dir, 'blob')
gabor_dir = os.path.join(aug_dir, 'gabor')
brief_dir = os.path.join(aug_dir, 'brief')


#%%
def get_files_of_sequence(seq):
  dir, id = os.path.split(seq)
  id = id.split('.')[0]

  dir, label = os.path.split(dir)  
  seq_path = os.path.join(frames_dir, '*' + '_'.join([label, id]) + '_*')

  frame_files = glob.glob(seq_path)
  return frame_files

def get_class_videos(label):
  return glob.glob(os.path.join(videos_dir, label, '*'))
  
def split_class_videos(videos):
  training_sequences = np.random.choice(videos, int(0.8 * len(videos)), replace=False).tolist()
  testing_sequences = list(set(videos) - set(training_sequences))
  
  return [training_sequences, testing_sequences]
    
def split_videos():
  classes = os.listdir(videos_dir)
  classes_videos = list(map(get_class_videos, classes))
  splits = list(map(split_class_videos, classes_videos))
  
  training_sequences = reduce(list.__add__, (map(lambda x: x[0], splits)))
  testing_sequences = reduce(list.__add__, map(lambda x: x[1], splits))
  
  return training_sequences, testing_sequences

def get_files_by_type():
  np.random.seed(213)  
  training_sequences, testing_sequences = split_videos()    
  training_files = reduce(list.__add__, list(map(get_files_of_sequence, training_sequences)))
  testing_files = reduce(list.__add__, list(map(get_files_of_sequence, testing_sequences)))
  
  print('Training size:', len(training_files), 'Testing size:', len(testing_files))
  return {'train': training_files, 'test': testing_files}
  
def get_all_files():
  files_by_type = get_files_by_type()
  all_files = reduce(list.__add__, files_by_type.values())
  return all_files


#%%  
def run_extractions():
  gabor_file = 'gabor_features.csv'
  gabor_isomap_file = 'gabor_iso_features.csv'

  '''
  print('Starting frame extraction')
  frame_extraction_manager = FrameExtractionManager(videos_dir=videos_dir, frames_dir=frames_dir, downsample=4)
  frame_extraction_manager.run_extraction()
  '''


  files = get_all_files()
  print('Starting Brief feature extraction')
  brief_extraction_manager = BriefExtractionManager(files, brief_dir)
  brief_extraction_manager.run_extraction()
  
  '''
  print('Starting Blob feature extraction')
  blob_extraction_manager = BlobExtractionManager(files, blob_dir)
  blob_extraction_manager.run_extraction()
  
  print('Starting Gabor feature extraction')  
  gabor_manager = GaborExtractionManager(files, gabor_dir, gabor_file)
  gabor_manager.run_extraction()
  
  print('Starting Isomap feature extraction')
  isomap_manager = ISOFeatureManager(gabor_dir, gabor_file, gabor_isomap_file, n_components=50)
  isomap_manager.run_extraction()
  '''
#%%

