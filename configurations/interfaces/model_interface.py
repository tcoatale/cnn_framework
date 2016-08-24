# -*- coding: utf-8 -*-
import configurations.models.alex.tiny_aug
import configurations.models.alex.tiny

import configurations.models.vgg.tiny
import configurations.models.vgg.normal

import configurations.models.resnet.tiny
import configurations.models.resnet.small_aug
import configurations.models.resnet.small
import configurations.models.resnet.normal

import configurations.models.inception_resnet.dual_tiny
import configurations.models.inception_resnet.tiny
import configurations.models.inception_resnet.tiny_aug_full

import configurations.models.basic

models_dictionnary = {
  'alex': {
    'tiny_aug': configurations.models.alex.tiny_aug,
    'tiny': configurations.models.alex.tiny
  }, 'vgg': {
    'tiny': configurations.models.vgg.tiny, 
    'normal': configurations.models.vgg.normal
  }, 'resnet': {
    'small_aug': configurations.models.resnet.small_aug,
    'tiny': configurations.models.resnet.tiny, 
    'small': configurations.models.resnet.small, 
    'normal': configurations.models.resnet.normal
  }, 'inception': {
    'tiny_aug_full': configurations.models.inception_resnet.tiny_aug_full,
    'dual_tiny': configurations.models.inception_resnet.dual_tiny, 
    'tiny': configurations.models.inception_resnet.tiny
  }, 'basic': {
    'normal': configurations.models.basic
  }
}

def get_model(name, size):
  try:
    return models_dictionnary[name][size]
  except KeyError:
    print('Sorry, this is not a correct model.')

