# -*- coding: utf-8 -*-
import configurations.training.fast
import configurations.training.slow

params_dictionnary = {'fast': configurations.training.fast, 
                      'slow' :configurations.training.slow}

def get_params(name):
  try:
    return params_dictionnary[name]  
  except KeyError:
    print('Sorry, this parameterization is not correct.')