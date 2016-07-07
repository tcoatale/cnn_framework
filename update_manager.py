from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class UpdateManager:
  def __init__(self, config):
    self.config = config
  
  def training_loss(self, logits, labels):
    training_loss = self.config.training_loss(logits, labels)
    l2_loss = tf.add_n(tf.get_collection('losses'))

    tf.scalar_summary('training_loss', training_loss)
    tf.scalar_summary('l2_loss', l2_loss)

    return tf.add(training_loss, l2_loss)
    
  def evaluation_loss(self, logits, labels):
    evaluation_loss = self.config.evaluation_loss(logits, labels)
    tf.scalar_summary('evaluation_loss', evaluation_loss)
    return evaluation_loss
  
  def update(self, loss, global_step):
    """Train model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      loss_total: Total loss from loss().
      global_step: Integer Variable counting the number of training steps processed.
    Returns:
      train_op: op for training.
    """
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
  
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.train.AdagradOptimizer(learning_rate, 0.01)
    train_op = optimizer.minimize(loss, global_step=global_step)
  
    return train_op