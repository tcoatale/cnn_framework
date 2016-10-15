# -*- coding: utf-8 -*-
from datasets.json_interface import JsonInterface
from datasets.dataset import Dataset
import numpy as np
from functools import reduce
import glob
import os


class PCLEDataset(Dataset):
  def __init__(self):
    interface = JsonInterface('datasets/pcle/metadata.json')
    self.data = interface.parse()
    self.class_dict = {'meningioma': 0, 'GBM': 1}

  def label_handler(self, file):
    _, file_name = os.path.split(file)
    class_name = file_name.split('_')[0]
    return self.class_dict[class_name]
    
  def get_files_of_sequence(self, seq):
    frames_dir = os.path.join(self.data['directories']['raw'], 'frames')

    dir, id = os.path.split(seq)
    id = id.split('.')[0]
    dir, label = os.path.split(dir)  
    seq_path = os.path.join(frames_dir, '*' + '_'.join([label, id]) + '_*')
  
    frame_files = glob.glob(seq_path)
    return frame_files
  
  def get_class_videos(self, label):
    videos_dir = os.path.join(self.data['directories']['raw'], 'videos')
    return glob.glob(os.path.join(videos_dir, label, '*'))
    
  def split_class_videos(self, videos):
    training_sequences = np.random.choice(videos, int(0.8 * len(videos)), replace=False).tolist()
    testing_sequences = list(set(videos) - set(training_sequences))
    
    return [training_sequences, testing_sequences]
      
  def split_videos(self):
    videos_dir = os.path.join(self.data['directories']['raw'], 'videos')
    
    classes = os.listdir(videos_dir)
    classes_videos = list(map(self.get_class_videos, classes))
    splits = list(map(self.split_class_videos, classes_videos))
    
    training_sequences = reduce(list.__add__, (map(lambda x: x[0], splits)))
    testing_sequences = reduce(list.__add__, map(lambda x: x[1], splits))
    
    return training_sequences, testing_sequences
  
  def get_files_by_type(self):
    np.random.seed(122)  
    training_sequences, testing_sequences = self.split_videos()    
    training_files = reduce(list.__add__, list(map(self.get_files_of_sequence, training_sequences)))
    testing_files = reduce(list.__add__, list(map(self.get_files_of_sequence, testing_sequences)))
    
    print('Training size:', len(training_files), 'Testing size:', len(testing_files))
    return {'train': training_files, 'test': testing_files}
    
  def get_all_files(self):
    files_by_type = self.get_files_by_type()
    all_files = reduce(list.__add__, files_by_type.values())
    return all_files
  
