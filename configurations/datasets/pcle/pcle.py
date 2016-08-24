# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#%% Dataset information
name = 'pcle'

set_sizes = {'train': 6000, 'test': 1200}

identifier_bytes=6
label_bytes=1
aug_feature_bytes=1152
classes=2

def split_labels(original_label):
  true_sparse_label = tf.cast(tf.transpose(tf.gather(tf.transpose(original_label), list(range(1)))), tf.int32)
  augmentation_sparse_label = tf.cast(tf.transpose(tf.gather(tf.transpose(original_label), list(range(1, 2)))), tf.int32)
    
  return augmentation_sparse_label, true_sparse_label
  
def sparse_to_dense_id(file_id_tensor):
  file_id_tensor_int32 = tf.cast(file_id_tensor, dtype=tf.int32)

  multiplier_np = 256**(np.linspace(identifier_bytes-1, 0, num=identifier_bytes))
  multiplier = tf.constant(multiplier_np)
  multiplier_int32 = tf.cast(multiplier, dtype=tf.int32)
  
  
  file_id = tf.reduce_sum(tf.mul(file_id_tensor_int32, multiplier_int32))
  return file_id
  
def retrieve_file_id(file_number):
  return 'img_' + str(file_number) + '.jpg'
