import os
import pandas as pd
from preprocessing.datasets.pn.metadata_parser import MetadataParser
from preprocessing.datasets.pn.segment_parser import SegmentParser

class SequenceParser:
  def __init__(self, sequence, app_raw_data_root):
    self.sequence = sequence
    self.app_raw_data_root = app_raw_data_root
        
  def parse_metadata(self):
    metadata_parser = MetadataParser(self.sequence, self.app_raw_data_root)
    metadata_parser.process_segmentation_data()
    self.segmentation_data = metadata_parser.segmentation_data

  def parse_segmentation_line(self, segmentation_line):
    segment_parser = SegmentParser(segmentation_line, self.app_raw_data_root)
    df = segment_parser.run()
    return [df]
    
  def get_full_dataset(self):    
    frames = self.segmentation_data.apply(self.parse_segmentation_line, axis=1)
    frames = frames.tolist()
    frames = list(map(lambda f: f[0], frames))
    full_df = pd.concat(frames)
    return full_df
      
  def parse_sequence(self):
    self.parse_metadata()
    path = os.path.join(self.app_raw_data_root, self.sequence + '.csv')
    full_df = self.get_full_dataset()
    full_df.to_csv(path, index=False)
    
#%%
'''
class Infix:
  def __init__(self, function):
      self.function = function
  def __ror__(self, other):
      return Infix(lambda x, self=self, other=other: self.function(other, x))
  def __or__(self, other):
      return self.function(other)
  def __rlshift__(self, other):
      return Infix(lambda x, self=self, other=other: self.function(other, x))
  def __rshift__(self, other):
      return self.function(other)
  def __call__(self, value1, value2):
      return self.function(value1, value2)
      
pip=Infix(lambda x,f: f(x))
#%%
def square(x):
  return x**2
  
def minusone(x):
  return x-1
  
def my_pip(x):
  return x |pip| square |pip| minusone

my_pip(2)
#%%
'''