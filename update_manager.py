from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class UpdateManager:
  def __init__(self, config):
    self.config = config
  
  def training_loss(self, logits, labels):
    training_loss = self.config.training_loss(logits, labels)
    tf.add_to_collection('losses', training_loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.add_to_collection('losses', total_loss)
    return total_loss
    
  def _histogram_grad(self, gradient):
    grad, var = gradient
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)
    
  
  def update(self, loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = self.config.dataset.train_size / self.config.training_params.batch_size
    decay_steps = int(num_batches_per_epoch * self.config.training_params.num_epochs_per_decay)
  
    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(self.config.training_params.initial_learning_rate,
                                              global_step,
                                              decay_steps,
                                              self.config.training_params.learning_rate_decay_factor,
                                              staircase=True)
                                    
    tf.scalar_summary('learning_rate', learning_rate)
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses)
    list(map(lambda l: tf.scalar_summary(l.op.name, loss_averages.average(l)), losses))
    
    with tf.control_dependencies([loss_averages_op]):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      grads = optimizer.compute_gradients(loss)
  
    # Apply gradients.
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    
    list(map(lambda var: tf.histogram_summary(var.op.name, var), tf.trainable_variables()))
    list(map(self._histogram_grad, grads))
        
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(self.config.training_params.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')
  
    return train_op