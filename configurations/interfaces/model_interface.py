# -*- coding: utf-8 -*-
import configurations.models.alex.tiny_aug_1
import configurations.models.alex.tiny_aug_2
import configurations.models.alex.tiny_aug_3
import configurations.models.alex.aug_4
import configurations.models.alex.aug_5
import configurations.models.alex.aug_6
import configurations.models.alex.tiny
import configurations.models.inception_resnet.standard
import configurations.models.resnet.standard
import configurations.models.resnet.aug_1

models_dictionnary = {
  'alex': {
    'tiny_aug_1': configurations.models.alex.tiny_aug_1,
    'tiny_aug_2': configurations.models.alex.tiny_aug_2,
    'tiny_aug_3': configurations.models.alex.tiny_aug_3,
    'aug_4': configurations.models.alex.aug_4,
    'aug_5': configurations.models.alex.aug_5,
    'aug_6': configurations.models.alex.aug_6,
    'tiny': configurations.models.alex.tiny
  }, 'inception': {
    'standard': configurations.models.inception_resnet.standard
  }, 'resnet': {
    'standard': configurations.models.resnet.standard,
    'aug_1': configurations.models.resnet.aug_1
  }
}

def get_model(name, size):
  try:
    return models_dictionnary[name][size]
  except KeyError:
    print('Sorry, this is not a correct model.')

