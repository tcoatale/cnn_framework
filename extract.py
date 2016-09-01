# -*- coding: utf-8 -*-
from preprocessing.interfaces.dataset_interface import get_dataset
dataset = get_dataset('pcle')
dataset.run_extractions()
