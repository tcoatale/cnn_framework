import os
import tensorflow as tf
from functools import reduce

name = 'cifar10'
size = 32
data_dir = os.path.join('data', 'processed', name)

#%% Dataset information
train_size=50000
valid_size=10000
label_bytes=1
original_shape=[size, size, 3]
classes=10

#%%
imsize=24
imshape=[imsize, imsize, 3]
n_input = reduce(int.__mul__, imshape)


#%%
def distort_inputs(reshaped_image):
  distorted_image = tf.random_crop(reshaped_image, imshape)
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
  float_image = tf.image.per_image_whitening(distorted_image)
  return float_image