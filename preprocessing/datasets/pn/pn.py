# -*coding: utf-8 -*-
import os
import glob
from preprocessing.datasets.pn.sequence_parser import SequenceParser
from functools import reduce
import pandas as pd

from preprocessing.extractors.gabor_extractor import GaborExtractionManager
from preprocessing.extractors.isomap_extractor import ISOFeatureManager
from preprocessing.extractors.hog_extractor import HogExtractionManager

app_raw_data_root = os.path.join('data', 'raw', 'pn')

aug_data_root = os.path.join('data', 'augmented', 'pn')
gabor_dir = os.path.join(aug_data_root, 'gabor')
hog_dir = os.path.join(aug_data_root, 'hog')

dest_dir_base = os.path.join('data', 'processed', 'pn')

training_sequences = ['20160707', '20160505', '20150505', '20151126', '20140619', '20140911']
testing_sequences = ['20160107']

class ParsingManager:
  def __init__(self, sequences, app_raw_data_root):
    self.sequences = sequences
    self.app_raw_data_root = app_raw_data_root
    
  def unparsed_sequence(self, seq):
    csv = os.path.join(self.app_raw_data_root, seq + '.csv')
    return not os.path.exists(csv)
  
  def parse_sequence(self, seq):
    parser = SequenceParser(seq, self.app_raw_data_root)
    parser.parse_sequence()
    
  def parse_sequences(self):
    unparsed_sequences = list(filter(self.unparsed_sequence, self.sequences))
    print('Sequences to be parsed:', unparsed_sequences)
    list(map(self.parse_sequence, unparsed_sequences))
    
  
def get_files_of_sequence(seq):
  files_of_seq_found = glob.glob(os.path.join(app_raw_data_root, 'frames', '*' + seq + '*'))
  df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(app_raw_data_root, '*.csv'))))
  files_in_df = df.files.unique().tolist()
  accepted_files = list(set(files_of_seq_found) & set(files_in_df))
  
  return accepted_files

def get_files_by_type():
  parsing_manager = ParsingManager(training_sequences + testing_sequences, app_raw_data_root)
  parsing_manager.parse_sequences()
  
  training_files = reduce(list.__add__, list(map(get_files_of_sequence, training_sequences)))
  testing_files = reduce(list.__add__, list(map(get_files_of_sequence, testing_sequences)))
  
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
  
