from configurations.models.layers import conv2d_layer, fc_layer, flat, readout_layer, softmax_layer, pool_layer, normalize

def architecture(input):  
  conv1 = conv2d_layer(input, [5, 5], 64, name='conv1')
  pool1 = normalize(pool_layer(conv1, 2, name='pool1'))
  conv2 = conv2d_layer(pool1, [3, 3], 64, name='conv2')
  pool2 = pool_layer(normalize(conv2), 2, name='pool2')
  return pool2
  
def output(input, training_params, dataset):
  reshape = flat(input)
  fc1 = fc_layer(reshape, 384, name='fc1')
  fc2 = fc_layer(fc1, 192, name='fc2')
  return softmax_layer(readout_layer(fc2, dataset.classes, name='out'))