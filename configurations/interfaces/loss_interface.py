# -*- coding: utf-8 -*-
import configurations.losses.driver
import configurations.losses.pn

losses_dictionnary = {'pn': configurations.losses.pn, 'driver': configurations.losses.driver}

def get_loss(name):
  try:
    return losses_dictionnary[name]
  except KeyError:
    print('Sorry, this is not a correct loss module.')

