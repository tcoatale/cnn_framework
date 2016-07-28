import tensorflow as tf

#%% Dataset information
name = 'pn'
train_size=38000
valid_size=17000
label_bytes=5
classes=36

def split_labels(original_label):
  true_labels = tf.transpose(tf.gather(tf.transpose(original_label), [label_bytes-1]))
  index = tf.transpose(tf.gather(tf.transpose(original_label), list(range(label_bytes-1))))
  return index, true_labels
