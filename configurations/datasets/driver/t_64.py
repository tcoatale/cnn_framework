import os
import tensorflow as tf
from functools import reduce

name = 'driver'
size = 64
data_dir = os.path.join('data', 'processed', name, str(size))

#%% Dataset information
train_size=14000
valid_size=8000
submission_size = 80000
label_bytes=2
id_bytes = 4
original_shape=[size, size, 3]
classes=10
sub_classes=2

#%%
classes_1=2
classes=10
imsize=64

imshape=[imsize, imsize, 3]
n_input = reduce(int.__mul__, imshape)

def distort_inputs(reshaped_image):
  float_image = tf.image.per_image_whitening(reshaped_image)
  return float_image


def split_labels(original_label):
  label2 = tf.cast(tf.div(original_label, 256), tf.int32)
  label1 = tf.sub(original_label, tf.mul(label2, 256))
  
  return label1, label2