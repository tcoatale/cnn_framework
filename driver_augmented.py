import tensorflow as tf
import os
from helper import conv_maxpool_norm, local_layer, softmax_layer, inception
from functools import reduce

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
batch_size=128
max_steps=100000
num_examples=2000
num_submission = 2000 
display_freq=10
summary_freq=100
valid_freq=10
save_freq=10000

#%%
classes_1=2
classes=10

imsize=192
imshape=[192, 192, 3]
moving_average_decay=0.9999
num_epochs_per_decay=65.0
learning_rate_decay_factor=0.7
initial_learning_rate=0.1

#%% Evaluation information
eval_interval_secs = 60 * 5
run_once=True

#%% Read information
eval_data='test' #Either test or train_eval
log_device_placement=False

#%%
n_input = reduce(int.__mul__, imshape)
keep_prob = 0.80

def combined_to_single_labels(original_label):
  label2 = tf.cast(tf.div(original_label, 256), tf.int32)
  label1 = tf.sub(original_label, tf.mul(label2, 256))
  
  return label1, label2

#%%

def inference(images):
  conv1 = conv_maxpool_norm([3, 3], 16, 2, images, 'conv1')
  conv2 = conv_maxpool_norm([5, 5], 32, 2, conv1, 'conv2')
  inception2 = inception([[3, 3], [1, 1]], 16, 2, conv2, 'inception_module1')  
  conv3 = conv_maxpool_norm([5, 5], 64, 2, inception2, 'conv3')
  inception4 = inception([[3, 3], [5, 5]], 48, 2, conv3, 'inception_module2')  
#  conv4 = conv_maxpool_norm([5, 5], 96, 4, conv3, 'conv4')
  
  dropout_layer = tf.nn.dropout(inception4, keep_prob)
  reshape = tf.reshape(dropout_layer, [batch_size, -1])
  local51 = local_layer(384, reshape, 'local51')
  local61 = local_layer(192, local51, 'local61')
  softmax_linear1 = softmax_layer(classes_1, local61, 'softmax_layer1')
  
  concat = tf.concat(1, [reshape, softmax_linear1], name='concat')
  local52 = local_layer(384, concat, 'local52')
  local62 = local_layer(192, local52, 'local62')
  softmax_linear2 = softmax_layer(classes, local62, 'softmax_layer2')
  
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
  
  dual_loss = tf.add(loss1, loss2)
  
  # Calculate the average cross entropy loss across the batch.
  tf.add_to_collection('losses', dual_loss)

  # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
  
def evaluation_loss(logits, labels):
  logits1, logits2 = logits
  labels1, labels2 = combined_to_single_labels(labels)
  loss1 = individual_loss(logits1, labels1)
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