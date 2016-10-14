# -*- coding: utf-8 -*-
from datasets.interface import get_dataset
from formatting.preprocessing_manager import PreprocessingManager

dataset = get_dataset('pcle')
preprocessing_manager = PreprocessingManager(dataset)
preprocessing_manager.run()
