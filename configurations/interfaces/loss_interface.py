# -*- coding: utf-8 -*-
import configurations.losses.driver
import configurations.losses.dual_driver
import configurations.losses.pn
import configurations.losses.cifar10

losses_dictionnary = {'pn': configurations.losses.pn, 
                      'driver': configurations.losses.driver, 
                      'dual_driver': configurations.losses.dual_driver, 
                      'cifar10': configurations.losses.cifar10}

def get_loss(name):
  try:
    return losses_dictionnary[name]
  except KeyError:
    print('Sorry, this is not a correct loss module.')

