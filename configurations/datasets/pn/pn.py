import tensorflow as tf

#%% Dataset information
name = 'pn'
train_size=23000
valid_size=17000
label_bytes=5
classes=36

def split_labels(original_label):
  true_labels = tf.transpose(tf.gather(tf.transpose(original_label), [label_bytes-1]))
  true_labels_int32 = tf.to_int32(true_labels)
  boolean_dense_labels = tf.equal(tf.range(0, classes), true_labels_int32)
  
  index = tf.transpose(tf.gather(tf.transpose(original_label), list(range(label_bytes-1))))
  return index, boolean_dense_labels
