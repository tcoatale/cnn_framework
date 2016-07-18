import configurations.pn_tiny_vgg
import configurations.pn_tiny_alex
import configurations.driver_tiny_vgg
import configurations.driver_tiny_alex
import configurations.pn_tiny_resnet

configurations_dictionnary = {'pn' : {
                                'tiny_vgg': configurations.pn_tiny_vgg, 
                                'tiny_alex': configurations.pn_tiny_alex,
                                'tiny_resnet': configurations.pn_tiny_resnet},
                              'driver' : {
                                'tiny_vgg': configurations.driver_tiny_vgg, 
                                'tiny_alex': configurations.driver_tiny_alex }
                             }

def get_config(dataset=None, model=None):
    return configurations_dictionnary[dataset][model].get_config()