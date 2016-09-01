import os
import glob
import pandas as pd

class SegmentParser:
  def __init__(self, segmentation_line, app_raw_data_root):
    self.segmentation_line = segmentation_line
    self.data_dir = os.path.join(app_raw_data_root, 'frames')
    
  def in_video_file(self, line):
    index, _ = line
    return index >= self.segmentation_line.start_index and index < self.segmentation_line.end_index
    
  def get_frame_with_index(self, f):
    return int(f.split('_')[-1][:-4]), f
    
  def get_correct_frames(self):
    return glob.glob(os.path.join(self.data_dir, '*' + self.segmentation_line.Filename + '*.jpg'))
    
  def true_index(self, line):
    index, frame = line
    true_index = index + int(self.segmentation_line.true_start_index)
    return true_index, frame
    
  def add_class_data(self, line):
    true_index, frame = line
    return true_index, self.segmentation_line['class'], frame
    
  def run(self):
    frames = self.get_correct_frames()
    frames_with_index = map(self.get_frame_with_index, frames)
    frames_of_segment = filter(self.in_video_file, frames_with_index)
    frames_with_true_index = list(map(self.true_index, frames_of_segment))
    
    if frames_with_true_index == []:
      return None
    else:
      sorted_data = sorted(list(map(self.add_class_data, frames_with_true_index)))
      indices, labels, frames = list(zip(*sorted_data))
      df = pd.DataFrame({'index': indices, 'labels': labels, 'files': frames})
    return df