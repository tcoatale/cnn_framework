# -*- coding: utf-8 -*-
import preprocessing.datasets.pcle
import preprocessing.datasets.pn.pn

datasets_dictionnary = {'pn':preprocessing.datasets.pn.pn,
                        'pcle':preprocessing.datasets.pcle}

def get_dataset(name):
  try:
    return datasets_dictionnary[name]
  except KeyError:
    print('Sorry, this is not a correct dataset module.')

