import tensorflow as tf
import numpy as np

#%% Dataset information
name = 'pn'

train_size=1800
valid_size=200

identifier_bytes = 8
label_bytes=1
aug_feature_bytes = 10
classes=36

def split_labels(original_label):
  true_labels = tf.transpose(tf.gather(tf.transpose(original_label), [label_bytes-1]))
  index = tf.transpose(tf.gather(tf.transpose(original_label), list(range(label_bytes-1))))
  return index, true_labels

def sparse_to_dense_id(file_id_tensor):
  file_id_tensor_int32 = tf.cast(file_id_tensor, dtype=tf.int32)

  multiplier_np = 256**(np.linspace(identifier_bytes-1, 0, num=identifier_bytes))
  multiplier = tf.constant(multiplier_np)
  multiplier_int32 = tf.cast(multiplier, dtype=tf.int32)
  
  
  file_id = tf.reduce_sum(tf.mul(file_id_tensor_int32, multiplier_int32))
  return file_id