import tensorflow as tf
import os
from helper import alexnet
from functools import reduce

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
num_examples=2000
num_submission = 2000 
display_freq=10
summary_freq=100
valid_freq=10
save_freq=1000

#%%
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
  
#%%
def inference(images):
  return alexnet(images, keep_prob, batch_size, classes)

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
  
def classification_rate(model, images, labels):
  # Build a Graph that computes the logits predictions from the inference model.
  logits = model.inference(images)
  # Calculate predictions.
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  return top_k_op