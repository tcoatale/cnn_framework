# -*- coding: utf-8 -*-
import configurations.models.alex.tiny
import configurations.models.alex.small
import configurations.models.alex.normal

import configurations.models.vgg.tiny
import configurations.models.vgg.normal

import configurations.models.resnet.tiny
import configurations.models.resnet.small
import configurations.models.resnet.normal

import configurations.models.inception_resnet.tiny
import configurations.models.inception_resnet.small
import configurations.models.inception_resnet.normal

models_dictionnary = {
  'alex': {
    'tiny': configurations.models.alex.tiny, 
    'small': configurations.models.alex.small, 
    'normal': configurations.models.alex.normal}, 
  'vgg': {
    'tiny': configurations.models.vgg.tiny, 
    'normal': configurations.models.vgg.normal}, 
  'resnet': {
    'tiny': configurations.models.resnet.tiny, 
    'small': configurations.models.resnet.small, 
    'normal': configurations.models.resnet.normal},
  'inception': {
    'tiny': configurations.models.inception_resnet.tiny, 
    'small': configurations.models.inception_resnet.small, 
    'normal': configurations.models.inception_resnet.normal}
}

def get_model(name, size):
  try:
    return models_dictionnary[name][size]
  except KeyError:
    print('Sorry, this is not a correct model.')

