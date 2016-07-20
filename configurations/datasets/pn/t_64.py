import os
import tensorflow as tf
from functools import reduce
from configurations.datasets.pn.pn import *

size = 64
original_shape=[192, 64, 3]
data_dir = os.path.join('data', 'processed', name, str(size))

imshape=[160, 50, 3]
n_input = reduce(int.__mul__, imshape)


#%%
def distort_inputs(distorted_image):
  float_image = tf.image.per_image_whitening(distorted_image)
  return float_image