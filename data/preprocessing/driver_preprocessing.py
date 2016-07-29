# -*- coding: utf-8 -*-
import os
from driver_managers import ImageManager
from preprocessing_manager import PreprocessingManager

#%%
resize = 64
batch_size = 4000
train_ratio = 0.75

data_dir = os.path.join('..', 'raw', 'driver', 'train')
submission_dir = os.path.join('..', 'raw', 'driver', 'test')
aug_dir = os.path.join('..', 'augmented', 'driver', 'hog')
dest_dir = os.path.join('..', 'processed', 'driver', str(resize))

if not os.path.isdir(dest_dir):
  os.mkdir(dest_dir)

image_manager = ImageManager((resize, resize), aug_dir)
preprocessing_manager = PreprocessingManager(data_dir, submission_dir, dest_dir, train_ratio, batch_size)
preprocessing_manager.run(image_manager)