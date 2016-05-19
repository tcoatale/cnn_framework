import os
import numpy as np
from PIL import Image
from random import shuffle

resize = 256
n = 10

def load_thumbnail(f):
    im = Image.open(f)
    im = im.resize((resize, resize), Image.ANTIALIAS)
    result = np.asarray(im, dtype="uint8")
    return result
    
def write_chunk(chunks, i):
    binary_data = np.array(map(lambda (l, i): [l] + i.ravel().tolist(), chunks[i])).ravel()
    newFileByteArray = bytearray(binary_data)
    newFile = open ('data/driver/data_batch_' + str(i), "wb")
    newFile.write(newFileByteArray)


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
                
    #filenames = filenames[0:100]
    shuffle(filenames)
                    
    labels = np.array(map(lambda f: int(f.split('/')[3][1]), filenames), dtype=np.uint8)
    images = np.array(map(load_thumbnail, filenames))
    
    data = zip(labels, images)
    chunks = [data[i:i+n] for i in xrange(0, len(data), n)]
    
    map(lambda i: write_chunk(chunks, i), range(len(chunks)))

#%%
preprocess_data()
