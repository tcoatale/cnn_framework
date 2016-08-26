# -*- coding: utf-8 -*-
import configurations.models.alex.tiny_aug_1
import configurations.models.alex.tiny_aug_2
import configurations.models.alex.tiny_aug_3
import configurations.models.alex.tiny
import configurations.models.inception_resnet.tiny_aug_1

models_dictionnary = {
  'alex': {
    'tiny_aug_1': configurations.models.alex.tiny_aug_1,
    'tiny_aug_2': configurations.models.alex.tiny_aug_2,
    'tiny_aug_3': configurations.models.alex.tiny_aug_3,
    'tiny': configurations.models.alex.tiny
  }, 'inception': {
    'tiny_aug_1': configurations.models.inception_resnet.tiny_aug_1
  }
}

def get_model(name, size):
  try:
    return models_dictionnary[name][size]
  except KeyError:
    print('Sorry, this is not a correct model.')

