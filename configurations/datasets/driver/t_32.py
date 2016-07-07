import os
import tensorflow as tf
from functools import reduce

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
classes_1=2
classes=10
imsize=32

imshape=[imsize, imsize, 3]
n_input = reduce(int.__mul__, imshape)

def distort_inputs(reshaped_image):
  float_image = tf.image.per_image_whitening(reshaped_image)
  return float_image


def split_labels(original_label):
  label2 = tf.transpose(tf.gather(tf.transpose(original_label), list(range(classes))))
  label1 = tf.transpose(tf.gather(tf.transpose(original_label), list(range(classes, classes + sub_classes))))

  return label1, label2