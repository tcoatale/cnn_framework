# -*- coding: utf-8 -*-
from datasets.interface import get_dataset
from extraction.extraction_manager import ExtractionManager

dataset = get_dataset('pcle')

extraction_manager = ExtractionManager(dataset)
extraction_manager.run()