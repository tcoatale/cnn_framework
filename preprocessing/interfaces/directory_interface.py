# -*- coding: utf-8 -*-
import preprocessing.directories.driver

directories_dictionnary = {'driver': preprocessing.directories.driver}

def get_directory(name):
  try:
    return directories_dictionnary[name]
  except KeyError:
    print('Sorry, this is not a correct directory module.')

