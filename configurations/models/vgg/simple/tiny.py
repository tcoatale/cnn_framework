from configurations.models.layers import conv2d_layer, fc_layer, flat, readout_layer, softmax_layer, pool_layer, normalize

def architecture(input):  
  conv1 = conv2d_layer(input, [5, 5], 64, name='conv1')
  pool2 = normalize(pool_layer(conv1, 2, name='pool2'))
  conv3 = conv2d_layer(pool2, [3, 3], 64, name='conv3')
  pool4 = normalize(pool_layer(conv3, 2, name='pool4'))
  conv5 = conv2d_layer(pool4, [3, 3], 128, name='conv5')
  pool6 = pool_layer(normalize(conv5), 2, name='pool6')
  return pool6
  
def output(input, training_params, dataset):
  reshape = flat(input)
  fc1 = fc_layer(reshape, 384, name='fc1')
  fc2 = fc_layer(fc1, 192, name='fc2')
  return softmax_layer(readout_layer(fc2, dataset.classes, name='out'))