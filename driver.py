import tensorflow as tf
import os
from helper import conv_maxpool_norm, local_layer, softmax_layer

application = 'driver'
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
label_bytes=1
original_shape=[256, 256, 3]

#%% Training information
batch_size=128
max_steps=100000
num_examples=20000
display_freq=10
summary_freq=100
save_freq=1000
classes=10
imsize=192
imshape=[192, 192, 3]
moving_average_decay=0.9999
num_epochs_per_decay=65.0
learning_rate_decay_factor=0.7
initial_learning_rate=0.1

#%% Evaluation information
eval_interval_secs = 60 * 5
run_once=False

#%% Read information
eval_data='test' #Either test or train_eval
log_device_placement=False

#%%
n_input = reduce(int.__mul__, imshape)
keep_prob = 0.80
  
#%%
def inference(images):
  conv1 = conv_maxpool_norm([3, 3, 3, 16], 2, images, 'conv1')
  conv2 = conv_maxpool_norm([5, 5, 16, 32], 2, conv1, 'conv2')
  conv3 = conv_maxpool_norm([5, 5, 32, 64], 2, conv2, 'conv3')
  conv4 = conv_maxpool_norm([5, 5, 64, 96], 4, conv3, 'conv4')
    
  
  dropout_layer = tf.nn.dropout(conv4, keep_prob)
  reshape = tf.reshape(dropout_layer, [batch_size, -1])
  local5 = local_layer(384, reshape, 'local5')
  local6 = local_layer(192, local5, 'local6')
  softmax_linear = softmax_layer(classes, local6, 'softmax_layer')
  return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
  
def distorted_inputs(reshaped_image):
  distorted_image = tf.random_crop(reshaped_image, [imsize, imsize, 3])
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
  float_image = tf.image.per_image_whitening(distorted_image)
  return float_image