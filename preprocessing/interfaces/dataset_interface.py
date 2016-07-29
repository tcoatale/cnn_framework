# -*- coding: utf-8 -*-
import preprocessing.managers.driver_managers

datasets_dictionnary = {'driver': preprocessing.managers.driver_managers}

def get_dataset_managers(name):
  try:
    return datasets_dictionnary[name]
  except KeyError:
    print('Sorry, this is not a correct dataset module.')

