# -*- coding: utf-8 -*-
import configurations.losses.simple
import configurations.losses.simple_split
import configurations.losses.dual

losses_dictionnary = {'simple': configurations.losses.simple, 
                      'simple_split': configurations.losses.simple_split, 
                      'dual': configurations.losses.dual}

def get_loss(name):
  try:
    return losses_dictionnary[name]
  except KeyError:
    print('Sorry, this is not a correct loss module.')

