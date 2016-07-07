from configurations.models.layers import conv2d_layer, fc_layer, flat, readout_layer, softmax_layer

def architecture(input):  
  conv0 = conv2d_layer(input, [3, 3], 16)
  conv1 = conv2d_layer(conv0, [3, 3], 32)
  conv2 = conv2d_layer(conv1, [3, 3], 64)
  conv3 = conv2d_layer(conv2, [3, 3], 96)
  reshape = flat(conv3)
  fc1 = fc_layer(reshape, 128)
  return fc1
  
def output(input, training_params, dataset):
  return softmax_layer(readout_layer(input, dataset.classes))