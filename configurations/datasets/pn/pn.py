import tensorflow as tf
import numpy as np

#%% Dataset information
name = 'pn'
train_size=23000
valid_size=17000
label_bytes=5
classes=36

def split_labels(original_label):
  label2 = tf.transpose(tf.gather(tf.transpose(original_label), [label_bytes]))
  label1 = tf.transpose(tf.gather(tf.transpose(original_label), list(range(1,label_bytes))))
  return label1, label2
