# -*- coding: utf-8 -*-
from datasets.pcle.pcle import PCLEDataset

dataset_dict = {'pcle': PCLEDataset}

def get_dataset(dataset):
  return dataset_dict[dataset]()
