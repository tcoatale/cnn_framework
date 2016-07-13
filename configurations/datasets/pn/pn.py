import tensorflow as tf

#%% Dataset information
name = 'pn'
train_size=23000
valid_size=17000
label_bytes=5
classes=36

def split_labels(original_label):
  true_labels = tf.transpose(tf.gather(tf.transpose(original_label), [label_bytes-1]))
  true_labels_int32 = tf.reshape(tf.to_int32(true_labels),[-1]) 
  #true_labels_dense = tf.to_int32(tf.equal(tf.range(0, classes), true_labels_int32))
  #true_labels_dense_float = tf.to_float(true_labels_dense)  
  index = tf.transpose(tf.gather(tf.transpose(original_label), list(range(label_bytes-1))))
  return index, true_labels_int32
