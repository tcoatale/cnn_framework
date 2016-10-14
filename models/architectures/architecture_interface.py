# -*- coding: utf-8 -*-
from models.architectures.alexnet import AlexNet, AlexNetFeatures, AlexNetChannels1, AlexNetFull1
from models.architectures.basic import Basic

architecture_dict = {
  'alex_standard': AlexNet,
  'alex_features': AlexNetFeatures,
  'alex_channels_1': AlexNetChannels1,
  'alex_full_1': AlexNetFull1,
  'basic': Basic
}

def get_architecture(architecture_name):
  return architecture_dict[architecture_name]()
