# -*- coding: utf-8 -*-
from preprocessing.extractors.gabor_extractor import GaborExtractionManager
from preprocessing.extractors.isomap_extractor import ISOFeatureManager
from preprocessing.extractors.hog_extractor import HogExtractionManager
from preprocessing.interfaces.directory_interface import get_directory

data_directory = get_directory('driver')

gabor_file = 'gabor_features.csv'
gabor_isomap_file = 'gabor_isomap_features.csv'
n_components=10

print('Starting Gabor feature extraction')
gabor_manager = GaborExtractionManager(data_directory.all_files, data_directory.gabor_dir, gabor_file)
gabor_manager.run_extraction()

'''
print('Starting Isomap feature extraction')
isomap_manager = ISOFeatureManager(data_directory.gabor_dir, gabor_file, gabor_isomap_file, n_components)
isomap_manager.run_extraction()

print('Starting Hog feature extraction')
hog_augmentation_manager = HogExtractionManager(image_files=data_directory.testing_image_files, #all_files, 
                                                  dest_dir=data_directory.hog_dir, 
                                                  pixels_per_cell=12, 
                                                  orientations=8)

hog_augmentation_manager.run_extraction()
'''
