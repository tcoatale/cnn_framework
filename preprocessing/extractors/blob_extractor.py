import skimage.io
import skimage.color
import os
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
import numpy as np
from functools import reduce
from datetime import datetime
import pathos.multiprocessing as mp

class BlobFeatureExtractor:
  def is_coorect_coord(self, coord, image_shape)  :
    x, y = coord
    w, h = image_shape
    
    correct_x = x >= 0 and x < w
    correct_y = y >= 0 and y < h
    
    return correct_x and correct_y
  
  def is_in_blob(self, coord, center, r):
    x, y = coord
    x_c, y_c = center
    return (x-x_c)**2 + (y-y_c)**2 < r**2


  def get_coordinates(self, blob):
    x, y, r = blob
    center = np.array([x, y], dtype=np.int16)
    x_min, y_min = center - int(r)
    x_max, y_max = center + int(r)
    
    x_range = range(x_min, x_max+1)
    y_range = range(y_min, y_max+1)
    
    coords = list(map(lambda x: list(map(lambda y: (x, y), y_range)), x_range))
    coords = reduce(list.__add__, coords)
    
    coords_blob = filter(lambda coord: self.is_in_blob(coord, center, r), coords)
    return coords_blob
  
  
  def get_blobs(self, image):
    blobs_log = blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    blobs_dog = blob_dog(image, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    blobs_doh = blob_doh(image, max_sigma=30, threshold=.01)
    
    all_blobs = np.vstack([blobs_log, blobs_doh, blobs_dog])
    all_blobs = filter(lambda b: b[2] > 4, all_blobs)
    all_blobs = list(filter(lambda b: b[2] < 60, all_blobs))
    return all_blobs
  
  
  def update_blobs(self, image, blob):
    coords_blob = self.get_coordinates(blob)    
    coords_blob = filter(lambda coord: self.is_coorect_coord(coord, image.shape), coords_blob)
    coords_blob = list(zip(*coords_blob))
    image[coords_blob] = 1  
  
  def extract(self, file):
    image = skimage.io.imread(file)
    blobs = self.get_blobs(image)
    blob_image = np.zeros(image.shape)
    list(map(lambda b: self.update_blobs(blob_image, b), blobs))
    
    final_blob_image = np.array(blob_image * 255, dtype = np.uint8)
    return final_blob_image


class BlobExtractionManager:
  def __init__(self, image_files, dest_dir):
    
    self.image_files = image_files
    self.dest_dir = dest_dir
    self.extractor = BlobFeatureExtractor()
    
  def compute_blobs(self, file):
    return self.extractor.extract(file)
    
  def extract(self, line):
    index, file = line

    if index % 50 == 0:
      print(datetime.now(), index)
    
    blob_image = self.compute_blobs(file)    
    _, id = os.path.split(file)
    dest_file = os.path.join(self.dest_dir, id)
    skimage.io.imsave(dest_file, blob_image)
    
  def split_in_batches(self, n_split):
    image_files = self.image_files
    n_images = len(image_files)
    indices = np.linspace(0, n_images, n_split+1).astype(np.int)
    batch_indices = zip(indices[:-1], indices[1:])
    batches_of_files = [image_files[b[0]:b[1]] for b in batch_indices]
    
    return batches_of_files
    
  def run_extraction_batch(self, files):
    list(map(self.extract, enumerate(files)))
    
  def run_extraction(self):    
    processes = 4
    batches_of_files = self.split_in_batches(processes)
    print('Split files in', len(batches_of_files), 'batches')
    pool = mp.ProcessingPool(processes)
    pool.map(lambda batch_files: self.run_extraction_batch(batch_files), batches_of_files)
