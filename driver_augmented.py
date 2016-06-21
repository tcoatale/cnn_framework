import tensorflow as tf
import os
from functools import reduce
from helper import residual_inception, reduction, local_layer, conv2d, softmax_layer, average_pool_output

application = 'driver_augmented'
log_dir = 'log'
eval_dir = 'eval'
ckpt_dir = 'ckpt'
data_dir = 'data'
raw_dir = 'raw'

#%% Directories
log_dir = os.path.join(log_dir, application)
eval_dir = os.path.join(eval_dir, application)
ckpt_dir = os.path.join(ckpt_dir, application)
data_dir = os.path.join(data_dir, application)
raw_dir = os.path.join(raw_dir, application)

#%% Dataset information
train_size=20000
valid_size=2000
label_bytes=2
id_bytes = 4
original_shape=[256, 256, 3]

#%% Training information
batch_size=64
max_steps=100000
num_examples=2000
num_submission = 2000 
display_freq=10
summary_freq=100
valid_freq=10
save_freq=1000

#%%
classes_1=2
classes=10

imsize=192
imshape=[192, 192, 3]
moving_average_decay=0.9999
num_epochs_per_decay=30.0
learning_rate_decay_factor=0.7
initial_learning_rate=0.1

#%% Evaluation information
eval_interval_secs = 60 * 30
run_once=False

#%% Read information
eval_data='test' #Either test or train_eval
log_device_placement=False

#%%
n_input = reduce(int.__mul__, imshape)
keep_prob = 0.70

def combined_to_single_labels(original_label):
  label2 = tf.cast(tf.div(original_label, 256), tf.int32)
  label1 = tf.sub(original_label, tf.mul(label2, 256))
  
  return label1, label2
#%%

def inference(images):
  conv1 = conv2d([7, 7], 24, images, 'conv1')
  print(conv1.get_shape())
  
  l2 = reduction(conv1, name='l2')
  l3 = residual_inception(l2, name='l3')
  l4 = residual_inception(l3, name='l4')
  l5 = reduction(l4, name='l5')
  l6 = residual_inception(l5, name='l6')
  l7 = reduction(l6, name='l7')
  l8 = residual_inception(l7, name='l8')
  l9 = reduction(l8, name='l9')
  l10 = residual_inception(l9, name='l10')
  
  
  print(conv1.get_shape())
  print(l2.get_shape())
  print(l3.get_shape())
  print(l4.get_shape())
  print(l5.get_shape())
  print(l6.get_shape())
  print(l7.get_shape())
  print(l8.get_shape())
  print(l9.get_shape())
  print(l10.get_shape())
      
  dropout_layer = tf.nn.dropout(l10, keep_prob)
  softmax_linear1 = average_pool_output([1, 1], classes_1, dropout_layer, 'softmax_layer1')

  reshape = tf.reshape(dropout_layer, [batch_size, -1])  
  concat = tf.concat(1, [reshape, softmax_linear1], name='concat')

  local10_2 = local_layer(192, concat, 'local10_2')
  softmax_linear2 = softmax_layer(classes, local10_2, 'softmax_layer2')
  
  return softmax_linear1, softmax_linear2

def individual_loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)  
  cross_entropy_mean = tf.reduce_mean(cross_entropy)

  return cross_entropy_mean

def loss(logits, labels):
  logits1, logits2 = logits
  labels1, labels2 = combined_to_single_labels(labels)
  
  loss1 = individual_loss(logits1, labels1)
  loss2 = individual_loss(logits2, labels2)
  
  dual_loss = tf.add(loss2, tf.mul(loss1, 0.5))
  
  # Calculate the average cross entropy loss across the batch.
  #tf.add_to_collection('losses', dual_loss)

  # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
  #return tf.add_n(tf.get_collection('losses'), name='total_loss')

  tf.add_to_collection('total_loss', dual_loss)
  return dual_loss
  
def evaluation_loss(logits, labels):
  logits1, logits2 = logits
  labels1, labels2 = combined_to_single_labels(labels)
  loss1 = individual_loss(logits2, labels2)
  return loss1

def classification_rate(model, images, labels):
  # Build a Graph that computes the logits predictions from the inference model.
  logits, _ = model.inference(images)
  labels, _ = combined_to_single_labels(labels)

  # Calculate predictions.
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  return top_k_op
  
def distorted_inputs(reshaped_image):
  distorted_image = tf.random_crop(reshaped_image, [imsize, imsize, 3])
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
  float_image = tf.image.per_image_whitening(distorted_image)
  return float_image