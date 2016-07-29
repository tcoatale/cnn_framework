import tensorflow as tf
import numpy as np
from configurations.datasets.helper import sparse_to_dense

#%% Dataset information
name = 'driver'

classes = 10
sub_classes = 2

train_size=14000
valid_size=8000
submission_size = 80000

identifier_bytes = 4
label_bytes=2

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
  
def retrieve_file_id(file_array_id):
  multiplier = list(map(lambda i: (2 ** 8) ** i, range(identifier_bytes)))
  multiplier.reverse()
  file_number = int(np.dot(file_array_id, multiplier))
  return 'img_' + str(file_number) + '.jpg'