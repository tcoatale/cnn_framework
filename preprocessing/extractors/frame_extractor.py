# -*- coding: utf-8 -*-
import os
import glob
import imageio
import skimage.io
import skimage.color

class FrameExtractor:
  def __init__(self, frames_dir, video_file, downsample):
    self.video_file = video_file
    self.frames_dir = frames_dir
    self.downsample = downsample
    
  def write(self, line):
    name, frame = line
    skimage.io.imsave(name, frame)
    
  def preprocess_frame(self, frame):
    return skimage.color.rgb2gray(frame)
    
  def run_extraction(self):
    dir, id = os.path.split(self.video_file)
    id, _ = id.split('.')
    dir, label = os.path.split(dir)
    
    reader = imageio.get_reader(self.video_file,  'ffmpeg', loop=False)
    frames = list(reader)
    frames = frames[::self.downsample]
    gray_frames = map(self.preprocess_frame, frames)
    
    names = map(lambda i: os.path.join(self.frames_dir, '_'.join([label, id, str(i) + '.jpg'])), range(len(frames)))
    results = zip(names, gray_frames)
    
    list(map(self.write_frame, results))
    
    
class FrameExtractionManager:
  def __init__(self, videos_dir, frames_dir, downsample):
    self.videos_dir = videos_dir
    self.frames_dir = frames_dir
    self.downsample = downsample
    
  def run_extraction_video(self, line):
    index, video = line
    print('Extracting frames from video', index) 
    extractor = FrameExtractor(self.frames_dir, video, self.downsample)
    extractor.run_extraction()

  def run_extraction(self):
    videos_path = os.path.join(self.videos_dir, '*', '*')
    videos = glob.glob(videos_path)
    list(map(self.run_extraction_video, enumerate(videos)))
