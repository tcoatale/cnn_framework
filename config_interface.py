import configurations.driver_tiny_vgg as configuration

def get_config():
    return configuration.get_config()
    
def get_config_by_name(name):
  return __import__('configurations.' + name).get_config()