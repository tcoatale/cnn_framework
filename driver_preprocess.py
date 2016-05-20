import os
import numpy as np
from PIL import Image
from random import shuffle

resize = 256
testing_size = 2000
images_per_file = 10000

def load_thumbnail(f):
    im = Image.open(f)
    im = im.resize((resize, resize), Image.ANTIALIAS)
    result = np.asarray(im, dtype="uint8")
    return result
    
def write_chunk(chunks, i, data_type):
    print('\t', 'chunk:', i)    
    binary_data = np.array(map(lambda (l, i): [l] + i.ravel().tolist(), chunks[i])).ravel()
    newFileByteArray = bytearray(binary_data)
    newFile = open ('data/driver/' + data_type + '_batch_' + str(i), "wb")
    newFile.write(newFileByteArray)
    
def write(filenames, data_type):
    labels = np.array(map(lambda f: int(f.split('/')[3][1]), filenames), dtype=np.uint8)
    images = np.array(map(load_thumbnail, filenames))
    
    data = zip(labels, images)
    chunks = [data[i:i+images_per_file] for i in xrange(0, len(data), images_per_file)]
    
    map(lambda i: write_chunk(chunks, i, data_type), range(len(chunks)))
    
def preprocess_data(): 
    unique_labels = os.listdir('raw/driver/train')
    dirs = map(lambda l: os.path.join('raw/driver/train', l), unique_labels)
    filenames = reduce(
                    list.__add__, 
                    map(
                        lambda dir: map(lambda f: os.path.join(dir, f), os.listdir(dir)), 
                        dirs
                    )
                )
                
    shuffle(filenames)
    
    test_filenames = filenames[:testing_size]
    train_filenames = filenames[testing_size:]
    
    print('Testing data')    
    write(test_filenames, 'test')
    print('Training data')    
    write(train_filenames, 'data')

#%%
preprocess_data()
