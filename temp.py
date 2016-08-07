import skimage.io
import skimage.color
import os
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
import numpy as np
from functools import reduce



class BlobFeatureExtractor:
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
    coords_blob = list(zip(*coords_blob))
  
  
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
    image[coords_blob] = 1  
  
  def extract(self, file):
    image = skimage.io.imread(file)
    blobs = self.get_blobs(image)
    
    blob_image = np.zeros(image.shape)
    list(map(lambda b: self.update_blobs(blob_image, b), blobs))
    
    final_blob_image = np.array(blob_image * 255, dtype = np.uint8)
    return final_blob_image


class BlobExtractionManager:
  def __init__(self, files, dest_dir):
    self.file = files
    self.dest_dir = dest_dir
    self.extractor = BlobFeatureExtractor()
    
  def compute_blobs(self, file):
    return self.extractor.extract(file)
    
  def extract(self, file):
    blob_image = self.compute_blobs(file)
    
    _, id = os.path.split(file)
    dest_file = os.path.join(self.dest_dir, id)
    skimage.io.imsave(dest_file, blob_image)