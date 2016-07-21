# -*- coding: utf-8 -*-
import configurations.training.fast
import configurations.training.slow
import configurations.training.slower
import configurations.training.extra_slow

params_dictionnary = {'fast': configurations.training.fast, 'slow' :configurations.training.slow, 'slower': configurations.training.slower, 'extra_slow': configurations.training.extra_slow}

def get_params(name):
  try:
    return params_dictionnary[name]  
  except KeyError:
    print('Sorry, this parameterization is not correct.')