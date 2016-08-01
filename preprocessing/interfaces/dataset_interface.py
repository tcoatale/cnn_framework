# -*- coding: utf-8 -*-
import preprocessing.datasets.driver
import preprocessing.datasets.pn.pn

datasets_dictionnary = {'driver': preprocessing.datasets.driver,
                        'pn':preprocessing.datasets.pn.pn}

def get_dataset(name):
  try:
    return datasets_dictionnary[name]
  except KeyError:
    print('Sorry, this is not a correct dataset module.')

