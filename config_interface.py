import configurations.pn_tiny_vgg
import configurations.pn_tiny_alex
import configurations.pn_normal_alex_32
import configurations.pn_normal_alex_64
import configurations.pn_tiny_resnet
import configurations.pn_small_resnet

import configurations.driver_tiny_vgg
import configurations.driver_tiny_alex
import configurations.driver_normal_alex

configurations_dictionnary = {'pn' : {
                                'tiny_vgg': configurations.pn_tiny_vgg, 
                                'tiny_alex': configurations.pn_tiny_alex,
                                'tiny_resnet': configurations.pn_tiny_resnet,
                                'small_resnet': configurations.pn_small_resnet,
                                'normal_alex_32': configurations.pn_normal_alex_32,
                                'normal_alex_64': configurations.pn_normal_alex_64},
                              'driver' : {
                                'tiny_vgg': configurations.driver_tiny_vgg, 
                                'tiny_alex': configurations.driver_tiny_alex,
                                'normal_alex': configurations.driver_normal_alex},
                             }

def get_config(dataset=None, model=None):
    return configurations_dictionnary[dataset][model].get_config()