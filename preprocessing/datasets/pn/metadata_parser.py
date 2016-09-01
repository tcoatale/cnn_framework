import os
import pandas as pd

class MetadataParser:
  def __init__(self, sequence, app_raw_data_root):
    self.subsampling_rate = 10
    self.sequence = sequence
    self.segmentation_data = pd.read_excel(os.path.join(app_raw_data_root, sequence + '.xlsx'))
    self.meta = pd.read_excel(os.path.join(app_raw_data_root, 'meta.xlsx'))
    self.classes = pd.read_excel(os.path.join(app_raw_data_root, 'classes.xlsx'))

  def get_class_from_desc(self, d):
    return self.classes[self.classes.Description == d].iloc[0].Classification
    
  def time_to_index(self, t, freq):
    return int((3600 * t.hour + 60 * t.minute + t.second) * freq / self.subsampling_rate)
  
  def get_freq_of_sequence(self):
    return self.meta[self.meta.Operation == int(self.sequence)].iloc[0].Frequency
    
  def get_file_start_index(self, line):
    file = line.Filename
    file_lines = self.segmentation_data[self.segmentation_data.Filename == file]
    start_file = file_lines.cum_index.min()
    line['true_start_index'] = line.start_index + start_file
    line['true_end_index'] = line.end_index + start_file
    return line
      
  def convert_times(self, metadata):
    freq = self.get_freq_of_sequence()  
    metadata['start_index'] = metadata.Start.apply(lambda t: self.time_to_index(t, freq))
    metadata['end_index'] = metadata.Finish.apply(lambda t: self.time_to_index(t, freq))

    metadata['n_frames'] = metadata.end_index - metadata.start_index    
    metadata['cum_index'] = metadata.n_frames.cumsum() - metadata.n_frames
    
    metadata = metadata.apply(self.get_file_start_index, axis=1)
    
    del metadata['Start']
    del metadata['Finish']
    del metadata['n_frames']
    del metadata['cum_index']
  
    return metadata
    
  def compute_class(self, metadata):
    metadata['class'] = metadata.Description.apply(lambda d: self.get_class_from_desc(d))
    del metadata['Description']
    return metadata

  def process_segmentation_data(self):
    metadata = self.segmentation_data
    metadata = self.convert_times(metadata)
    metadata = self.compute_class(metadata)
    del metadata['Left Instrument']
    del metadata['Right Instrument']
    
    self.segmentation_data = metadata