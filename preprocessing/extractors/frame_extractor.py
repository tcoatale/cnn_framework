# -*- coding: utf-8 -*-
import os
import glob
import imageio
import skimage.io
import skimage.color


videos_dir = os.path.join('data/pcle/raw/videos')
frames_dir = os.path.join('data/pcle/raw/frames')

videos_path = os.path.join(videos_dir, '*', '*')
videos = glob.glob(videos_path)

def preprocess_frame(frame):
  return skimage.color.rgb2gray(frame)

def write_frame(line):
  name, frame = line
  skimage.io.imsave(name, frame)

def get_frames(video_file):
  dir, id = os.path.split(video_file)
  id, _ = id.split('.')
  dir, label = os.path.split(dir)
  
  reader = imageio.get_reader(video_file,  'ffmpeg', loop=False)
  frames = list(reader)
  gray_frames = map(preprocess_frame, frames)
  
  names = map(lambda i: os.path.join(frames_dir, '_'.join([label, id, str(i) + '.jpg'])), range(len(frames)))
  results = zip(names, gray_frames)
  
  list(map(write_frame, results))  
  

#%%
list(map(get_frames, videos))