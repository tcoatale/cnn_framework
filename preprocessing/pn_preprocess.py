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
training_sequences = ['20160707']
testing_sequences = ['20160505']
meta = pd.read_excel(os.path.join(dataset_dir, 'meta.xlsx'))
classes = pd.read_excel(os.path.join(dataset_dir, 'classes.xlsx'))





  


  




def four_bytes_index(index):
    return list(struct.unpack('4B', struct.pack('>I', index)))
  
  

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
#sequence = testing_sequences[0]
#write_sequence_in_chunks('test_batch', sequence, meta, classes)

