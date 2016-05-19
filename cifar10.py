import tensorflow as tf
import os
from helper import _activation_summary, _variable_with_weight_decay, _variable_on_cpu

FLAGS = tf.app.flags.FLAGS

application = 'cifar10'
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
tf.app.flags.DEFINE_integer('train_size', 50000, """Training images""")
tf.app.flags.DEFINE_integer('valid_size', 10000, """a""")
tf.app.flags.DEFINE_integer('label_bytes', 1, """a""")
tf.app.flags.DEFINE_integer('original_shape', [32, 32, 3], """a""")

#%% Training information
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_examples', 10000, """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('display_freq', 10, """q""")
tf.app.flags.DEFINE_boolean('summary_freq', 100, """q""")
tf.app.flags.DEFINE_boolean('save_freq', 100, """q""")
tf.app.flags.DEFINE_integer('classes', 10, """a""")
tf.app.flags.DEFINE_integer('imsize', 24, """a""")
tf.app.flags.DEFINE_integer('imshape', [24, 24, 3], """a""")

#%% Evaluation information
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False, """Whether to run eval only once.""")

#%% Read information
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

#%%
n_input = reduce(int.__mul__, FLAGS.imshape)

out_conv_1 = 32
out_conv_2 = 32
    
n_hidden_1 = 384
n_hidden_2 = 192

dropout = 0.90
NUM_CLASSES = FLAGS.classes

#%%
def inference(images):
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.nn.softmax(tf.add(tf.matmul(local4, weights), biases, name=scope.name))
    _activation_summary(softmax_linear)

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

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
  
def distorted_inputs(reshaped_image):
  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(reshaped_image, [FLAGS.imsize, FLAGS.imsize, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)
  
  return float_image