import os
import tensorflow as tf
from functools import reduce

name = 'driver_augmented'
data_dir = os.path.join('data', name)

#%% Dataset information
train_size=14000
valid_size=8000
label_bytes=2
id_bytes = 4
original_shape=[256, 256, 3]
classes=10
sub_classes=2

#%%
eval_freq = 300
display_freq=10
summary_freq=50
valid_freq=10
save_freq=1000

#%%
classes_1=2
classes=10
imsize=192
imshape=[192, 192, 3]
n_input = reduce(int.__mul__, imshape)

def distort_inputs(reshaped_image):
  distorted_image = tf.random_crop(reshaped_image, [imsize, imsize, 3])
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=200)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=3.8)
  float_image = tf.image.per_image_whitening(distorted_image)
  return float_image


def split_labels(original_label):
  label2 = tf.cast(tf.div(original_label, 256), tf.int32)
  label1 = tf.sub(original_label, tf.mul(label2, 256))
  
  return label1, label2