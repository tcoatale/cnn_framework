import tensorflow as tf
import os
from functools import reduce
import datetime
from helper import alexnet, vggnet, resnet, inception_resnet

application = 'driver_augmented'
log_dir = 'log'
eval_dir = 'eval'
ckpt_dir = 'ckpt'
data_dir = 'data'
raw_dir = 'raw'

#%% Directories
def date_to_suffix(date):
    return '-'.join(list(map(str, [date.year, date.month, date.day, date.hour, date.minute, date.second])))

suffix = date_to_suffix(datetime.datetime.now())

log_dir = os.path.join(log_dir, application, suffix)
eval_dir = os.path.join(eval_dir, application, suffix)
ckpt_dir = os.path.join(ckpt_dir, application, suffix)

list(map(os.mkdir, [log_dir, eval_dir, ckpt_dir]))

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
summary_freq=50
valid_freq=10
save_freq=1000

#%%
classes_1=2
classes=10

imsize=192
imshape=[192, 192, 3]
moving_average_decay=0.9999
num_epochs_per_decay=5.0
learning_rate_decay_factor=0.7
initial_learning_rate=0.05

#%% Evaluation information
eval_interval_secs = 60 * 30
run_once=False

#%% Read information
eval_data='test' #Either test or train_eval
log_device_placement=False

#%%
n_input = reduce(int.__mul__, imshape)
keep_prob = 0.7

def combined_to_single_labels(original_label):
  label2 = tf.cast(tf.div(original_label, 256), tf.int32)
  label1 = tf.sub(original_label, tf.mul(label2, 256))
  
  return label1, label2
#%%

def inference(images):
  #tf.image_summary('images', images)

  softmax_linear2 = inception_resnet(images, keep_prob, classes)
  #softmax_linear2 = vggnet(images, keep_prob, batch_size, classes)
  #return softmax_linear1, softmax_linear2
  return softmax_linear2

def individual_loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)  
  cross_entropy_mean = tf.reduce_mean(cross_entropy)

  return cross_entropy_mean

def loss(logits, labels):
  '''
  logits1, logits2 = logits
  
  loss1 = individual_loss(logits1, labels1)
  loss2 = individual_loss(logits2, labels2)
  
  dual_loss = tf.add(loss2, tf.mul(loss1, 0.5))
  '''
  labels1, labels2 = combined_to_single_labels(labels)
  dual_loss = individual_loss(logits, labels2)
  tf.add_to_collection('losses', dual_loss)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
  
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
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=200)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=3.8)
  float_image = tf.image.per_image_whitening(distorted_image)
  return float_image