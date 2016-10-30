# -*- coding: utf-8 -*-
from models.architectures.building_blocks.output_blocks import fc_output
from models.architectures.building_blocks.layers import conv2d_layer, pool_layer, flat, fc_layer
import tensorflow as tf

class AlexNet:
  def _start(self, input, add_filters, features):
    print(input.get_shape())
    conv1 = conv2d_layer(input, [11, 11], 96, name='conv1')
    pool2 = pool_layer(conv1, 2, name='pool2')
    conv3 = conv2d_layer(pool2, [5, 5], 128, name='conv3')
    pool4 = pool_layer(conv3, 2, name='pool4')
    conv5 = conv2d_layer(pool4, [3, 3], 256, name='conv5')
    return conv5
  
  def _end(self, input, add_filters, features):
    conv6 = conv2d_layer(input, [3, 3], 256, name='conv6')
    conv7 = conv2d_layer(conv6, [3, 3], 128, name='conv7')
    pool8 = pool_layer(conv7, 2, name='pool8')
    print(pool8.get_shape())
    return pool8
  
  def _output(self, input, dataset):  
    return fc_output(input, dataset.data['classes'], [128], name='output')    
    
class AlexNetFeatures(AlexNet):
  def _end(self, input, add_filters, features):
    conv6 = conv2d_layer(input, [3, 3], 256, name='conv6')
    conv7 = conv2d_layer(conv6, [3, 3], 128, name='conv7')
    pool8 = pool_layer(conv7, 2, name='pool8')
    print(pool8.get_shape())
    
    flat_features = flat(pool8)
    dropout_features = tf.nn.dropout(features, 0.8)
    full_features = tf.concat(1, [flat_features, dropout_features])
    units = 1024
    output_features = fc_layer(input=full_features, units=units, name='output_features')
    return output_features
    
class AlexNetChannels0(AlexNet):
  def _start(self, input, add_filters, features):
    augmented_image = tf.concat(3, [input, add_filters])
    print(augmented_image.get_shape())
    
    conv1 = conv2d_layer(input, [11, 11], 96, name='conv1')
    pool2 = pool_layer(conv1, 2, name='pool2')
    conv3 = conv2d_layer(pool2, [5, 5], 128, name='conv3')
    pool4 = pool_layer(conv3 , 2, name='pool4')
    conv5 = conv2d_layer(pool4, [3, 3], 256, name='conv5')
    return conv5
        
class AlexNetChannels1(AlexNet):
  def _start(self, input, add_filters, features):
    print(input.get_shape())
    conv1 = conv2d_layer(input, [11, 11], 96, name='conv1')
    augmented_image = tf.concat(3, [conv1, add_filters])
    pool2 = pool_layer(augmented_image, 2, name='pool2')
    conv3 = conv2d_layer(pool2, [5, 5], 128, name='conv3')
    pool4 = pool_layer(conv3, 2, name='pool4')
    conv5 = conv2d_layer(pool4, [3, 3], 256, name='conv5')
    return conv5

class AlexNetChannels2(AlexNet):
  def _start(self, input, add_filters, features):
    print(input.get_shape())
    conv1 = conv2d_layer(input, [11, 11], 96, name='conv1')
    pool2 = pool_layer(conv1, 2, name='pool2')
    conv3 = conv2d_layer(pool2, [5, 5], 128, name='conv3')
    pool_additional_filters = pool_layer(add_filters, 2, name='pool_additional_filters')
    augmented_image = tf.concat(3, [conv3, pool_additional_filters])
    pool4 = pool_layer(augmented_image, 2, name='pool4')
    conv5 = conv2d_layer(pool4, [3, 3], 256, name='conv5')
    return conv5
    
class AlexNetFull1(AlexNetChannels1):
  def _end(self, input, add_filters, features):
    conv6 = conv2d_layer(input, [3, 3], 256, name='conv6')
    conv7 = conv2d_layer(conv6, [3, 3], 128, name='conv7')
    pool8 = pool_layer(conv7, 2, name='pool8')
    print(pool8.get_shape())
    
    flat_features = flat(pool8)
    dropout_features = tf.nn.dropout(features, 0.8)
    full_features = tf.concat(1, [flat_features, dropout_features])
    units = 1024
    output_features = fc_layer(input=full_features, units=units, name='output_features')
    return output_features
    
class AlexNetFull2(AlexNetChannels2):
  def _end(self, input, add_filters, features):
    conv6 = conv2d_layer(input, [3, 3], 256, name='conv6')
    conv7 = conv2d_layer(conv6, [3, 3], 128, name='conv7')
    pool8 = pool_layer(conv7, 2, name='pool8')
    print(pool8.get_shape())
    
    flat_features = flat(pool8)
    dropout_features = tf.nn.dropout(features, 0.8)
    full_features = tf.concat(1, [flat_features, dropout_features])
    units = 1024
    output_features = fc_layer(input=full_features, units=units, name='output_features')
    return output_features
