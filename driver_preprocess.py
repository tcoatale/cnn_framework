import os
import numpy as np
from PIL import Image
from random import shuffle
from functools import reduce

resize = 256
testing_size = 8000
images_per_file = 4000

def load_thumbnail(f):
  im = Image.open(f)
  im = im.resize((resize, resize), Image.ANTIALIAS)
  result = np.asarray(im, dtype=np.uint8)
  return result
  
def get_labels(filenames):
    return np.array(list(map(lambda f: [int(f.split('/')[3][1])], filenames)), dtype=np.uint8)
    
def get_augmented_labels(filenames):
    labels = get_labels(filenames)
    augmentation = np.array(list(map(lambda l: [int(l in range(1, 5))], labels)), dtype=np.uint8)
    augmented_labels = np.hstack([labels, augmentation])
    return augmented_labels
    
def write_chunk(chunks, i, data_type, aug=False):
  print(data_type, '\t', 'chunk:', i)
  labels = get_labels(chunks[i]) if aug == False else get_augmented_labels(chunks[i])
  images = np.array(list(map(load_thumbnail, chunks[i])))
  name = 'data/driver_augmented/' + data_type + '_batch_' + str(i)
  write_data(labels, images, name)
  
def write_data(labels, images, name):
  labels = list(map(lambda l: l.tolist(), labels))
  images = list(map(lambda i: i.ravel().tolist(), images))
  labels_and_images = zip(labels, images)
  binary_data = list(map(lambda label_image: label_image[0] + label_image[1], labels_and_images))
  binary_data = np.array(binary_data).ravel()
  newFileByteArray = bytearray(binary_data)
  binary_data = None
  images = None
  newFile = open (name, "wb")
  newFile.write(newFileByteArray)
    
def write_in_chunks(filenames, data_type, aug=False):
  chunks = [filenames[i:i+images_per_file] for i in range(0, len(filenames), images_per_file)]
  list(map(lambda i: write_chunk(chunks, i, data_type, aug), range(len(chunks))))
    
def preprocess_data(): 
  unique_labels = os.listdir('raw/driver/train')
  dirs = list(map(lambda l: os.path.join('raw/driver/train', l), unique_labels))
  filenames = reduce(
                list.__add__, 
                list(map(
                  lambda dir: list(map(lambda f: os.path.join(dir, f), os.listdir(dir))), 
                  dirs
                ))
              )
                
  shuffle(filenames)

  test_filenames = filenames[:testing_size]
  train_filenames = filenames[testing_size:]
  write_in_chunks(test_filenames, 'test', True)
  write_in_chunks(train_filenames, 'data', True)

#%%
preprocess_data()