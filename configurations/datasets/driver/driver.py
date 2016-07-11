import tensorflow as tf
import numpy as np

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
  label2 = tf.transpose(tf.gather(tf.transpose(original_label), list(range(classes))))
  label1 = tf.transpose(tf.gather(tf.transpose(original_label), list(range(classes, classes + sub_classes))))
  return label1, label2
  
def retrieve_file_id(file_array_id):
  multiplier = list(map(lambda i: (2 ** 8) ** i, range(id_bytes)))
  multiplier.reverse()
  file_number = int(np.dot(file_array_id, multiplier))
  return 'img_' + str(file_number) + '.jpg'