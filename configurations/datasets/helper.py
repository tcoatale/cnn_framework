# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import PIL.Image

def sparse_to_dense(sparse_tensor, classes):
  sparse_labels = tf.reshape(sparse_tensor, [-1, 1])
  derived_size = tf.shape(sparse_tensor)[0]
  indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
  concated = tf.concat(1, [indices, sparse_labels])
  outshape = tf.pack([derived_size, classes])
  dense_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
  
  return dense_labels
  
def display(image):
  image = image - np.min(image)
  image = image * 255 / np.max(image)
  im = PIL.Image.fromarray(np.uint8(image))
  im.show()