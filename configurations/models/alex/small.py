from configurations.models.alex.alex import architecture as base_architecture
from configurations.models.blocks.output_blocks import fc_output

#%%
def architecture(input):
  return base_architecture(input)
  
def output(input, dataset):
  return fc_output(input, dataset.classes, [512, 256])
