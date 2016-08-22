from configurations.models.alex.alex import architecture as base_architecture
from configurations.models.blocks.output_blocks import fc_output

#%%
def architecture(image, add_filters):
  return base_architecture(image)
  
def output(input, features, dataset):  
  return fc_output(input, dataset.classes, [192, 128], name='output')