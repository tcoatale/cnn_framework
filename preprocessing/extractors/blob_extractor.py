import skimage.io
import skimage.color
import os
import numpy as np
from functools import reduce
from datetime import datetime
import pathos.multiprocessing as mp
import skimage.exposure
import glob
import cv2


def white_params():
  params = cv2.SimpleBlobDetector_Params()

  params.blobColor = 255
  
  params.minThreshold = 0
  params.maxThreshold = 500
  
  params.minArea = 40
  params.maxArea = 50000
  params.filterByArea = True
   
  params.minCircularity = 0.50
  params.filterByCircularity = True
   
  params.minConvexity = 0.50
  params.filterByConvexity = True
   
  params.filterByInertia = True
  params.minInertiaRatio = 0.3
  
  return params

def black_params():
  params = cv2.SimpleBlobDetector_Params()

  params.blobColor = 0
  
  params.minThreshold = 0
  params.maxThreshold = 100
  
  params.minArea = 400
  params.maxArea = 50000
  params.filterByArea = True
   
  params.minCircularity = 0.002
  params.filterByCircularity = True
   
  params.minConvexity = 0.02
  params.filterByConvexity = True
   
  params.filterByInertia = True
  params.minInertiaRatio = 0.003  
  return params  

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
    x, y = blob.pt
    r = blob.size

    center = np.array([x, y], dtype=np.int16)
    x_min, y_min = center - int(r)
    x_max, y_max = center + int(r)
    
    x_range = range(x_min, x_max+1)
    y_range = range(y_min, y_max+1)
    
    coords = list(map(lambda x: list(map(lambda y: (x, y), y_range)), x_range))
    coords = reduce(list.__add__, coords)
    
    coords_blob = filter(lambda coord: self.is_in_blob(coord, center, r), coords)
    return coords_blob
  
  
  def get_blobs(self, image, params):
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
       
    keypoints = filter(lambda b: b.size > 3, keypoints)
    keypoints = list(filter(lambda b: b.size < 100, keypoints))
    return keypoints
  
  
  def update_blobs(self, image, blob):
    coords_blob = self.get_coordinates(blob)    
    coords_blob = filter(lambda coord: self.is_coorect_coord(coord, image.shape), coords_blob)
    coords_blob = list(zip(*coords_blob))
    image[coords_blob] = 1  
  
  def extract(self, image, params):
    blobs = self.get_blobs(image, params)
    blob_image = np.zeros(image.shape)
    list(map(lambda b: self.update_blobs(blob_image, b), blobs))
    
    final_blob_image = np.array(blob_image * 255, dtype = np.uint8)
    return final_blob_image


class BlobExtractionManager:
  def __init__(self, frames_dir, dest_dir):
    self.frames = glob.glob(os.path.join(frames_dir, '*'))
    self.dest_dir = dest_dir
    self.extractor = BlobFeatureExtractor()
    self.white_parameters = white_params()
    self.black_parameters = black_params()
      
  def extract(self, line):
    index, file = line

    if index % 50 == 0:
      print(datetime.now(), index)
      
    image = skimage.io.imread(file)
    image = np.array(skimage.exposure.equalize_adapthist(image, clip_limit=0.03) * 255, dtype=np.uint8)

    white_blob_image = self.extractor.extract(image, self.white_parameters)
    black_blob_image = self.extractor.extract(image, self.black_parameters)
    
    _, id = os.path.split(file)
    
    id, ext = id.split('.')
    dest_file = os.path.join(self.dest_dir, id)
    
    skimage.io.imsave(dest_file + '_white.' + ext, white_blob_image)
    skimage.io.imsave(dest_file + '_black.' + ext, black_blob_image)
    
  def split_in_batches(self, n_split):
    image_files = self.frames
    n_images = len(image_files)
    indices = np.linspace(0, n_images, n_split+1).astype(np.int)
    batch_indices = zip(indices[:-1], indices[1:])
    batches_of_files = [image_files[b[0]:b[1]] for b in batch_indices]
    
    return batches_of_files
    
  def run_extraction_batch(self, files):
    list(map(self.extract, enumerate(files)))
    
  def run_extraction(self, parallel=False):
    processes = 4
    if parallel:
      batches_of_files = self.split_in_batches(processes)
      print('Split files in', len(batches_of_files), 'batches')
      pool = mp.ProcessingPool(processes)
      pool.map(lambda batch_files: self.run_extraction_batch(batch_files), batches_of_files)
      
    else:
      batches_of_files = self.split_in_batches(processes)
      list(map(lambda batch_files: self.run_extraction_batch(batch_files), batches_of_files))