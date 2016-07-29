import os
import tensorflow as tf
from functools import reduce
from configurations.datasets.driver.driver import *

size = 64
original_shape=[size, size, 4]
image_bytes = reduce(int.__mul__, original_shape)
data_dir = os.path.join('data', 'processed', name, str(size))

imsize=48
imshape=[imsize, imsize, 4]
n_input = reduce(int.__mul__, imshape)


#%%
def process_inputs(image, distort=True):
  if distort:
    image = tf.random_crop(image, imshape)
    image = tf.image.random_brightness(image, max_delta=65)
    image = tf.image.random_contrast(image, lower=0.20, upper=1.80)
  else:
    width, height, _ = imshape
    image = tf.image.resize_image_with_crop_or_pad(image, width, height)

  float_image = tf.image.per_image_whitening(image)
  return float_image