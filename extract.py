# -*- coding: utf-8 -*-
from datasets.json_interface import JsonInterface
from preprocessing.extractors.extraction_manager import ExtractionManager

interface = JsonInterface('datasets/pcle/metadata.json')
data = interface.parse()
extraction_manager = ExtractionManager(data)
extraction_manager.run()