# -*- coding: utf-8 -*-
import os

from preprocessing.extractors.frame_extractor import FrameExtractionManager
from preprocessing.extractors.gabor_extractor import GaborExtractionManager
from preprocessing.extractors.blob_extractor import BlobExtractionManager
from preprocessing.extractors.hog_extractor import HogExtractionManager
from preprocessing.extractors.brief_extractor import BriefExtractionManager
from preprocessing.extractors.isomap_extractor import ISOFeatureManager

extractor_dir = {
  'frames': FrameExtractionManager, 
  'gabor': GaborExtractionManager, 
  'blob': BlobExtractionManager, 
  'hog': HogExtractionManager, 
  'brief': BriefExtractionManager, 
  'isomap': ISOFeatureManager
}

class AugmentationExtractor:
  def __init__(self, data, augmentation):
    self.data = data    
    self.name = augmentation['name']
    self.priority = augmentation['priority']
    self.input = augmentation['input']
    self.output = augmentation['output']
    
    
  def _input(self):
    if self.input == 'videos':
      input = self.data['directories']['videos']
    elif self.input == 'frames':
      input = self.data['directories']['frames']
    else:
      input = os.path.join(self.data['directories']['augmentations'], self.input, 'output.csv')
      
    return input
    
  def _output(self):
    if self.input != 'frames':
      output_dir = os.path.join(self.data['directories']['augmentations'], self.input)
    else:
      output_dir = os.path.join(self.data['directories']['augmentations'], self.name)
      
    if self.output == 'frames':
      output = output_dir
    else:
      output = os.path.join(output_dir, 'output.csv')
    
    return output
      
  def _extractor(self):
    return extractor_dir[self.name]
    
  def run(self):
    input = self._input()
    output = self._output()
    extractor_class = self._extractor()

    print('Starting extraction:', self.name)    
    extractor = extractor_class(input, output)
    extractor.run_extraction()
    
  def __gt__(self, other):
    return self.priority > other.priority
    
  def __str__(self):
    return self.name
    
class ExtractionManager:
  def __init__(self, data):
    self.data = data
    
  def create(self, data, augmentation):
    return AugmentationExtractor(data, augmentation)
    
  def run(self):
    augmentations = data['augmentations']

    extractors = list(map(lambda augmentation: self.create(self.data, augmentation), augmentations))
    list(map(lambda e: e.run(), extractors))
    
#%%
from datasets.json_interface import JsonInterface
interface = JsonInterface('datasets/pcle/metadata.json')
data = interface.parse()

#%%
extraction_manager = ExtractionManager(data)
extraction_manager.run()