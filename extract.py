# -*- coding: utf-8 -*-
from preprocessing.extractors.gabor_extractor import GaborExtractionManager
from preprocessing.extractors.isomap_extractor import ISOFeatureManager
from preprocessing.extractors.hog_extractor import HogExtractionManager
from preprocessing.interfaces.dataset_interface import get_dataset

dataset = get_dataset('driver')

gabor_file = 'gabor_features.csv'
gabor_isomap_file = 'gabor_isomap_features.csv'
n_components=10

print('Starting Gabor feature extraction')
gabor_manager = GaborExtractionManager(dataset.get_all_files(), dataset.gabor_dir, gabor_file)
gabor_manager.run_extraction()

print('Starting Isomap feature extraction')
isomap_manager = ISOFeatureManager(dataset.gabor_dir, gabor_file, gabor_isomap_file, n_components)
isomap_manager.run_extraction()

print('Starting Hog feature extraction')
hog_augmentation_manager = HogExtractionManager(image_files=dataset.get_all_files(), 
                                                  dest_dir=dataset.hog_dir, 
                                                  pixels_per_cell=12, 
                                                  orientations=8)

hog_augmentation_manager.run_extraction()

