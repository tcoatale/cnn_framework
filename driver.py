import tensorflow as tf
import os
from helper import conv_maxpool_norm, local_layer, softmax_layer

FLAGS = tf.app.flags.FLAGS

application = 'driver'
log_dir = 'log'
eval_dir = 'eval'
ckpt_dir = 'ckpt'
data_dir = 'data'
raw_dir = 'raw'

#%% Directories
tf.app.flags.DEFINE_string('log_dir', os.path.join(log_dir, application), """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', os.path.join(eval_dir, application), """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('ckpt_dir', os.path.join(ckpt_dir, application), """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('data_dir', os.path.join(data_dir, application), """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('raw_dir', os.path.join(raw_dir, application), """Path to the CIFAR-10 data directory.""")

#%% Dataset information
tf.app.flags.DEFINE_integer('train_size', 20000, """Training images""")
tf.app.flags.DEFINE_integer('valid_size', 2000, """a""")
tf.app.flags.DEFINE_integer('label_bytes', 1, """a""")
tf.app.flags.DEFINE_integer('original_shape', [256, 256, 3], """a""")

#%% Training information
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 20000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_examples', 30000, """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('display_freq', 10, """q""")
tf.app.flags.DEFINE_boolean('summary_freq', 100, """q""")
tf.app.flags.DEFINE_boolean('save_freq', 100, """q""")
tf.app.flags.DEFINE_integer('classes', 10, """a""")
tf.app.flags.DEFINE_integer('imsize', 192, """a""")
tf.app.flags.DEFINE_integer('imshape', [192, 192, 3], """a""")
tf.app.flags.DEFINE_integer('moving_average_decay', 0.9999, """a""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 65.0, """a""")
tf.app.flags.DEFINE_integer('learning_rate_decay_factor', 0.7, """a""")
tf.app.flags.DEFINE_integer('initial_learning_rate', 0.1, """a""")

#%% Evaluation information
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False, """Whether to run eval only once.""")

#%% Read information
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

#%%
n_input = reduce(int.__mul__, FLAGS.imshape)
keep_prob = 0.80
  
#%%
def inference(images):
  conv1 = conv_maxpool_norm([3, 3, 3, 16], 2, images, 'conv1')
  conv2 = conv_maxpool_norm([5, 5, 16, 32], 2, conv1, 'conv2')
  conv3 = conv_maxpool_norm([5, 5, 32, 64], 2, conv2, 'conv3')
  conv4 = conv_maxpool_norm([5, 5, 64, 96], 4, conv3, 'conv4')
    
  
  dropout_layer = tf.nn.dropout(conv4, keep_prob)
  reshape = tf.reshape(dropout_layer, [FLAGS.batch_size, -1])
  local5 = local_layer(384, reshape, 'local5')
  local6 = local_layer(192, local5, 'local6')
  softmax_linear = softmax_layer(FLAGS.classes, local6, 'softmax_layer')
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
  distorted_image = tf.random_crop(reshaped_image, [FLAGS.imsize, FLAGS.imsize, 3])
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
  float_image = tf.image.per_image_whitening(distorted_image)
  return float_image