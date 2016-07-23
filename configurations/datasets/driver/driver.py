import tensorflow as tf
import numpy as np
from configurations.datasets.helper import sparse_to_dense

#%% Dataset information
name = 'driver'
train_size=14000
valid_size=8000
submission_size = 80000
label_bytes=2
id_bytes = 4
classes=10
sub_classes=2

def split_labels(original_label):
  true_sparse_label = tf.cast(tf.transpose(tf.gather(tf.transpose(original_label), list(range(1)))), tf.int32)
  augmentation_sparse_label = tf.cast(tf.transpose(tf.gather(tf.transpose(original_label), list(range(1, 2)))), tf.int32)
  
  true_dense_label = sparse_to_dense(true_sparse_label, classes)
  augmentation_dense_label = sparse_to_dense(augmentation_sparse_label, sub_classes)
  
  return augmentation_dense_label, true_dense_label
  
def retrieve_file_id(file_array_id):
  multiplier = list(map(lambda i: (2 ** 8) ** i, range(id_bytes)))
  multiplier.reverse()
  file_number = int(np.dot(file_array_id, multiplier))
  return 'img_' + str(file_number) + '.jpg'