# -*- coding: utf-8 -*-
import configurations.models.alex.tiny_aug_1
import configurations.models.alex.tiny_aug_2
import configurations.models.alex.tiny_aug_3
import configurations.models.alex.tiny
import configurations.models.inception_resnet.standard
import configurations.models.resnet.standard

models_dictionnary = {
  'alex': {
    'tiny_aug_1': configurations.models.alex.tiny_aug_1,
    'tiny_aug_2': configurations.models.alex.tiny_aug_2,
    'tiny_aug_3': configurations.models.alex.tiny_aug_3,
    'tiny': configurations.models.alex.tiny
  }, 'inception': {
    'standard': configurations.models.inception_resnet.standard
  }, 'resnet': {
    'standard': configurations.models.resnet.standard
  }
}

def get_model(name, size):
  try:
    return models_dictionnary[name][size]
  except KeyError:
    print('Sorry, this is not a correct model.')

