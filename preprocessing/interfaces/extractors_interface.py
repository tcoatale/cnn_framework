# -*- coding: utf-8 -*-
from preprocessing.extractors.gabor_extractor import GaborExtractionManager
from preprocessing.extractors.isomap_extractor import ISOFeatureManager
from preprocessing.extractors.hog_extractor import HogExtractionManager

extractors_dict = {
  '':, GaborExtractionManager
  '':, ISOFeatureManager
  '':, HogExtractionManager
}
