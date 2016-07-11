import os
import tensorflow as tf
from functools import reduce
from configurations.datasets.pn.pn import *

size = 32
original_shape=[96, 32, 3]
data_dir = os.path.join('data', 'processed', name, str(size))

imshape=[64, 24, 3]
n_input = reduce(int.__mul__, imshape)


#%%
def distort_inputs(distorted_image):
  distorted_image = tf.random_crop(distorted_image, imshape)
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=65)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.20, upper=1.80)
  float_image = tf.image.per_image_whitening(distorted_image)
  return float_image