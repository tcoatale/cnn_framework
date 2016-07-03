from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import _input
import config_interface

config = config_interface.get_config()

def distorted_inputs():
  return _input.distorted_inputs(data_dir=config.dataset.data_dir, batch_size=config.dataset.batch_size)

def inputs(data_type):
  return _input.inputs(data_type=data_type, data_dir=config.dataset.data_dir, batch_size=config.dataset.batch_size)
  
def loss_wrapper(logits, labels):
  training_loss = config.training_loss(logits, labels)
  tf.scalar_summary('training_loss', training_loss)
  tf.add_to_collection('losses', training_loss)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')  
    
def evaluation_loss(logits, labels):
  return config.evaluation_loss(logits, labels)

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  for l in losses + [total_loss]:
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def update(total_loss, global_step):
  """Train model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = config.dataset.train_size / config.training_params.batch_size
  decay_steps = int(num_batches_per_epoch * config.training_params.num_epochs_per_decay)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(config.training_params.initial_learning_rate,
                                  global_step,
                                  decay_steps,
                                  config.training_params.learning_rate_decay_factor,
                                  staircase=True)
                                  
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(config.training_params.moving_average_decay, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    update_op = tf.no_op(name='train')

  return update_op