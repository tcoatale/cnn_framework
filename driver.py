import tensorflow as tf
import os
import helper

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
tf.app.flags.DEFINE_integer('max_steps', 15000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_examples', 20000, """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('display_freq', 10, """q""")
tf.app.flags.DEFINE_boolean('summary_freq', 100, """q""")
tf.app.flags.DEFINE_boolean('save_freq', 100, """q""")
tf.app.flags.DEFINE_integer('classes', 10, """a""")
tf.app.flags.DEFINE_integer('imsize', 192, """a""")
tf.app.flags.DEFINE_integer('imshape', [192, 192, 3], """a""")

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
out_conv_3 = 32
out_conv_4 = 32
    
n_hidden_1 = 384
n_hidden_2 = 192

dropout = 0.90
NUM_CLASSES = FLAGS.classes

#%%
def inference(images):
  """Build the CIFAR model up to where it may be used for inference.
  Args:
  Returns:
    logits: Output tensor with the   computed logits.
  """

  # Reshape input picture
  print('In Inference ', images.get_shape(), type(images))
  images = tf.reshape(images, shape=[FLAGS.batch_size] + FLAGS.imshape)

  _dropout = tf.Variable(dropout)  # dropout (keep probability)

  # Store layers weight & bias
  _weights = {
    'wc1': helper.weight_variable([11, 11, 3, out_conv_1]),  # 5x5 conv, 3 input, 64 outputs
    'wc2': helper.weight_variable([7, 7, out_conv_1, out_conv_2]),
    'wc3': helper.weight_variable([5, 5, out_conv_2, out_conv_3]),
    'wc4': helper.weight_variable([3, 3, out_conv_3, out_conv_4]),
    'wd1': helper.weight_variable([out_conv_4 * 12 * 12, n_hidden_1]),
    'wd2': helper.weight_variable([n_hidden_1, n_hidden_2]),
    'out': helper.weight_variable([n_hidden_2, NUM_CLASSES])
  }

  _biases = {
    'bc1': helper.bias_variable([out_conv_1]),
    'bc2': helper.bias_variable([out_conv_2]),
    'bc3': helper.bias_variable([out_conv_3]),
    'bc4': helper.bias_variable([out_conv_4]),
    'bd1': helper.bias_variable([n_hidden_1]),
    'bd2': helper.bias_variable([n_hidden_2]),
    'out': helper.bias_variable([NUM_CLASSES])
  }
  
  with tf.name_scope('Conv1'):
    conv1 = helper.conv2d(images, _weights['wc1'], _biases['bc1'])
    conv1 = helper.max_pool(conv1, 2)
    conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    helper._activation_summary(conv1)
    
  with tf.name_scope('Conv2'):
    conv2 = helper.conv2d(conv1, _weights['wc2'], _biases['bc2'])
    conv2 = helper.max_pool(conv2, 2)
    conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    helper._activation_summary(conv2)
    
  with tf.name_scope('Conv3'):
    conv3 = helper.conv2d(conv2, _weights['wc3'], _biases['bc3'])
    conv3 = helper.max_pool(conv3, 2)
    conv3 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    helper._activation_summary(conv3)
    
  with tf.name_scope('Conv4'):
    conv4 = helper.conv2d(conv3, _weights['wc4'], _biases['bc4'])
    conv4 = helper.max_pool(conv4, 4)
    conv4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    helper._activation_summary(conv4)

  with tf.name_scope('Dense1'):
    dense1 = tf.reshape(conv4, [-1, _weights['wd1'].get_shape().as_list()[0]])  # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu_layer(dense1, _weights['wd1'], _biases['bd1'])  # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout
    helper._activation_summary(dense1)

  with tf.name_scope('Dense2'):
    dense2 = tf.nn.relu_layer(dense1, _weights['wd2'], _biases['bd2'])  # Relu activation
    helper._activation_summary(dense2)

  # Output, class prediction
  logits = tf.nn.softmax(tf.matmul(dense2, _weights['out']) + _biases['out'])
  helper._activation_summary(logits)

  return logits


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