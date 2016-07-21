# -*- coding: utf-8 -*-
import configurations.datasets.driver.t_32
import configurations.datasets.driver.t_64
import configurations.datasets.driver.t_96
import configurations.datasets.driver.t_128

import configurations.datasets.pn.t_32
import configurations.datasets.pn.t_64


datasets_dictionnary = {'pn': {'32': configurations.datasets.pn.t_32,
                             '64': configurations.datasets.pn.t_64}, 
                      'driver': {'32': configurations.datasets.driver.t_32, 
                                 '64': configurations.datasets.driver.t_64, 
                                 '96': configurations.datasets.driver.t_96, 
                                 '128': configurations.datasets.driver.t_128}}

def get_dataset(name, size):
  try:
    return datasets_dictionnary[name][size]
  except KeyError:
    print('Sorry, this is not a correct dataset module.')

