# -*- coding: utf-8 -*-
import configurations.datasets.driver.t_64
import configurations.datasets.pn.t_64
import configurations.datasets.pcle.t_64
import configurations.datasets.cifar10.cifar10


datasets_dictionnary = {'pn': {'64': configurations.datasets.pn.t_64}, 
                      'driver': {'64': configurations.datasets.driver.t_64},
                      'pcle': {'64': configurations.datasets.pcle.t_64},
                      'cifar10': {'32': configurations.datasets.cifar10.cifar10}}

def get_dataset(name, size):
  try:
    return datasets_dictionnary[name][size]
  except KeyError:
    print('Sorry, this is not a correct dataset module.')

