import os
import glob
import numpy as np
import PIL
import PIL.Image
from random import shuffle
from functools import reduce
import pandas as pd
import struct

resize = 48
subsampling_rate = 10
width = 192
height = 64
images_per_file = 5000

#%%
dataset_dir = os.path.join('..', 'raw', 'pn')
frames_dir = os.path.join(dataset_dir, 'frames')
training_frames_dir = os.path.join(frames_dir, 'train')
testing_frames_dir = os.path.join(frames_dir, 'test')
processed_dir = os.path.join('..', 'processed', 'pn')

#%%
training_sequences = ['20160107']
testing_sequences = ['20160505']
meta = pd.read_excel(os.path.join(dataset_dir, 'meta.xlsx'))
classes = pd.read_excel(os.path.join(dataset_dir, 'classes.xlsx'))


def get_frame_with_index(f):
  return int(f.split('_')[-1][:-4]), f

def in_range(i, start, end):
  return i >= start and i < end
  
def get_frames(l):
  frames_of_video = glob.glob(os.path.join(frames_dir, '*' + l.Filename + '*.jpg'))
  frames_with_index = map(get_frame_with_index, frames_of_video)
  frames_of_segment = filter(lambda f: in_range(f[0], l.start_index, l.end_index), frames_with_index)
  frames_with_true_index = map(lambda f: (f[0] + int(l.true_start_index), f[1]), frames_of_segment)
  final_data = sorted(list(map(lambda f: [f[0], l['class'], f[1]], frames_with_true_index)))
  return final_data
  
def dense_to_one_hot(labels, n_classes=2):
  """Convert class labels from scalars to one-hot vectors."""
  labels = np.array(labels)
  n_labels = labels.shape[0]
  index_offset = np.arange(n_labels) * n_classes
  labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
  labels_one_hot.flat[index_offset + labels.ravel()] = 1
  return labels_one_hot
  
def time_to_index(t, freq):
  return int((3600 * t.hour + 60 * t.minute + t.second) * freq / subsampling_rate)
  
def get_freq_of_sequence(sequence, meta):
  return meta[meta.Operation == int(sequence)].iloc[0].Frequency

def get_class_from_desc(d, classes):
  return classes[classes.Description == d].iloc[0].Classification

def four_bytes_index(index):
    return list(struct.unpack('4B', struct.pack('>I', index)))
  
def process_segmentation_data(segmentation, meta, classes):
  del segmentation['Left Instrument']
  del segmentation['Right Instrument']
  
  freq = get_freq_of_sequence(sequence, meta)

  segmentation['start_index'] = segmentation.Start.apply(lambda t: time_to_index(t, freq))
  segmentation['end_index'] = segmentation.Finish.apply(lambda t: time_to_index(t, freq))
  segmentation['n_frames'] = segmentation.end_index - segmentation.start_index
  segmentation['true_end_index'] = segmentation.end_index.cumsum()
  segmentation['true_start_index'] = segmentation.true_end_index - segmentation.n_frames
  
  del segmentation['Start']
  del segmentation['Finish']
  del segmentation['n_frames']
  
  segmentation['class'] = segmentation.Description.apply(lambda d: get_class_from_desc(d, classes))
  del segmentation['Description']
  
  return segmentation
  

def load_thumbnail(f):
  im = PIL.Image.open(f)
  im = im.resize((width, height), PIL.Image.ANTIALIAS)
  #im.show()
  result = np.asarray(im, dtype=np.uint8)
  result = np.transpose(result, [2, 0, 1])
  result = result.reshape(-1).tolist()
  
  return result
  
def process_line(f):
  index = four_bytes_index(f[0])
  label = f[1]
  image = load_thumbnail(f[2])
  
  return index + [label] + image
  
def parse_sequence(sequence, meta, classes):
  segmentation = pd.read_excel(os.path.join(dataset_dir, sequence + '.xlsx'))
  segmentation = process_segmentation_data(segmentation, meta, classes)
  frames = segmentation.apply(get_frames, axis=1).tolist()
  frames = reduce(list.__add__, frames)
  
  return frames
  
def write_chunk(sequence_type, sequence, chunks, i):
  print(sequence_type, sequence, i)
  chunk = list(map(process_line, chunks[i]))
  chunk = reduce(list.__add__, chunk)
  byte_array = bytearray(chunk)
  print('\tLoading done')
  file_name = '_'.join([sequence_type, str(sequence), str(i)])
  with open (os.path.join(processed_dir, file_name), "wb") as f:
    f.write(byte_array)
  print('\tWriting done')

def write_sequence_in_chunks(sequence_type, sequence, meta, classes):
  data = parse_sequence(sequence, meta, classes)
  chunks = [data[i:i+images_per_file] for i in range(0, len(data), images_per_file)]
  list(map(lambda i: write_chunk(sequence_type, sequence, chunks, i), range(len(chunks))))
  
#%%
sequence = training_sequences[0]
write_sequence_in_chunks('data_batch', sequence, meta, classes)
sequence = testing_sequences[0]
#write_sequence_in_chunks('test_batch', sequence, meta, classes)

