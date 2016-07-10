from configurations.models.layers import conv2d_layer, fc_layer, flat, readout_layer, softmax_layer, pool_layer

def architecture(input):  
  conv0 = conv2d_layer(input, [5, 5], 64)
  pool0 = pool_layer(conv0, 2)
  conv1 = conv2d_layer(pool0, [3, 3], 64)
  pool1 = pool_layer(conv1, 2)
  reshape = flat(pool1)
  fc1 = fc_layer(reshape, 384)
  fc2 = fc_layer(fc1, 192)
  return fc2
  
def output(input, training_params, dataset):
  return softmax_layer(readout_layer(input, dataset.classes))