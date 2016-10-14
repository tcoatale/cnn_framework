from models.architectures.building_blocks.layers import conv2d_layer, pool_layer, normalize
from models.architectures.building_blocks.output_blocks import fc_output

#%%
class Basic:
  def _start(self, input, add_filters, features):
    print(input.get_shape())
    conv1 = conv2d_layer(input, [5, 5], 64, name='conv1')
    pool2 = pool_layer(normalize(conv1), 2, name='pool2')
    return pool2
    
  def _end(self, input, add_filters, features):
    conv3 = conv2d_layer(input, [3, 3], 64, name='conv3')
    pool4 = normalize(pool_layer(conv3, 2, name='pool4'))
    print(pool4.get_shape())
    return pool4
  
  
  def _output(self, input, dataset):  
    return fc_output(input, dataset.data['classes'], [394, 192], name='output')