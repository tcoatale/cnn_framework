# -*- coding: utf-8 -*-
import os
import glob

app_raw_data_root = os.path.join('data', 'raw', 'driver')

train_data_dir = os.path.join(app_raw_data_root, 'train')
test_data_dir = os.path.join(app_raw_data_root, 'test')

training_image_files = glob.glob(os.path.join(train_data_dir, '*', '*'))
testing_image_files = glob.glob(os.path.join(test_data_dir, '*'))

all_files = training_image_files + testing_image_files
all_files = all_files

gabor_dir = os.path.join('data', 'augmented', 'driver', 'gabor')
hog_dir = os.path.join('data', 'augmented', 'driver', 'hog')