import os
import tensorflow as tf
from functools import reduce
import numpy as np

name = 'driver'
size = 32
data_dir = os.path.join('data', 'processed', name, str(size))

#%% Dataset information
train_size=14000
valid_size=8000
submission_size = 80000
label_bytes=12
id_bytes = 4
original_shape=[size, size, 3]
classes=10
sub_classes=2

#%%
imsize=32
imshape=[imsize, imsize, 3]
n_input = reduce(int.__mul__, imshape)

def distort_inputs(distorted_image):
  distorted_image = tf.random_crop(distorted_image, imshape)
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=65)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.20, upper=1.80)
  float_image = tf.image.per_image_whitening(distorted_image)
  return float_image


def split_labels(original_label):
  label2 = tf.transpose(tf.gather(tf.transpose(original_label), list(range(classes))))
  label1 = tf.transpose(tf.gather(tf.transpose(original_label), list(range(classes, classes + sub_classes))))

  return label1, label2
  
def retrieve_file_id(file_array_id):
  multiplier = list(map(lambda i: (2 ** 8) ** i, range(id_bytes)))
  multiplier.reverse()
  file_number = int(np.dot(file_array_id, multiplier))
  return 'img_' + str(file_number) + '.jpg'
  
  
  