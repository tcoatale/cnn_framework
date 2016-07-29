# -*- coding: utf-8 -*-
import os
import glob
from skimage.feature import hog
from skimage import color, exposure
from skimage import io
from datetime import datetime
  
#%%
class HogAugmentationManager:
  def __init__(self, files, dest_dir, pixels_per_cell, orientations):
    self.files = files
    self.dest_dir = dest_dir
    self.pixels_per_cell = pixels_per_cell
    self.orientations = orientations

  def compute_hog(self, files, index):
    if index % 50 == 0:
      print(datetime.now(), index)
    
    file = files[index]
    image = color.rgb2gray(io.imread(file))
    _, hog_image = hog(image, 
                       orientations=self.orientations, 
                       pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                       cells_per_block=(1, 1), 
                       visualise=True)
                       
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    #io.imshow(hog_image_rescaled)
    dest_file = os.path.join(self.dest_dir, os.path.split(file)[-1])
    io.imsave(dest_file, hog_image_rescaled)
    
  def run(self):
    list(map(lambda index: self.compute_hog(self.files, index), range(len(self.files))))
    
#%%
    
train_data_dir = os.path.join('..', 'raw', 'driver', 'train')
test_data_dir = os.path.join('..', 'raw', 'driver', 'test')
dest_dir = os.path.join('..', 'augmented', 'driver', 'hog')
training_image_files = glob.glob(os.path.join(train_data_dir, '*', '*'))
testing_image_files = glob.glob(os.path.join(test_data_dir, '*'))

hog_augmentation_manager = HogAugmentationManager(files=testing_image_files, 
                                                  dest_dir=dest_dir, 
                                                  pixels_per_cell=12, 
                                                  orientations=8)

hog_augmentation_manager.run()