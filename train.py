from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
from time import gmtime, strftime

from update_manager import UpdateManager
from input_manager import InputManager
import configurations.interfaces.configuration_interface as config_interface
import skimage.io

def train(config):
  """Train model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    input_manager = InputManager(config)
    update_manager = UpdateManager(config)

    # Get images and labels for dataset.
    with tf.variable_scope("training_inputs") as scope:
      training_ids, training_labels, training_images, training_add_filters, training_features = input_manager.get_inputs()
    with tf.variable_scope("eval_inputs") as scope:
      eval_ids,  eval_labels,  eval_images,  eval_add_filters,  eval_features = input_manager.get_inputs(type='test', distorted = False, shuffle = True)
    
    tf.image_summary('images', training_images, max_images=64)

    # Build a Graph that computes the logits predictions from the inference model.
    with tf.variable_scope("inference") as scope:
      training_logits = config.inference(training_images, training_add_filters, training_features)
      scope.reuse_variables()
      eval_logits = config.inference(eval_images, eval_add_filters, eval_features, testing=True)

    # Calculate loss.
    loss_training = config.training_loss(training_logits, training_labels)
    loss_eval = config.evaluation_loss(eval_logits, eval_labels)
    tf.scalar_summary('loss_eval', loss_eval)

    total_loss = update_manager.training_loss(loss_training)

    classirate_training = config.loss.classirate(config.dataset, training_logits, training_labels)
    classirate_eval = config.loss.classirate(config.dataset, eval_logits, eval_labels)    

    tf.scalar_summary('classirate_training', classirate_training)
    tf.scalar_summary('classirate_eval', classirate_eval)
    
   
    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = update_manager.update(total_loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.train.SummaryWriter(config.log_dir, sess.graph)

    for step in xrange(config.training_params.max_steps):
      step_increment = global_step.assign(step)
      start_time = time.time()

      labels, logits = sess.run([training_labels, training_logits])
      train_loss_value, eval_loss_value = sess.run([loss_training, loss_eval])
      total_loss_value = sess.run([total_loss])[0]      
      sess.run([step_increment, train_op])
      duration = time.time() - start_time

      assert not np.isnan(total_loss_value), 'Model diverged with loss = NaN'

      if step % config.display_freq == 0:
        num_examples_per_step = config.training_params.batch_size
        examples_per_sec = num_examples_per_step / duration

        print(strftime("%D %H:%M:%S", gmtime()), end=' ')
        print('Step', '%06d' % step, end=' ')
        print('Speed:', "%04d" % int(examples_per_sec), end=' ')
        print('Training loss:', total_loss_value, end=' ')
        print('T:', train_loss_value, end=' ')
        print('E:', eval_loss_value, end='\n')
        
      if step % config.summary_freq == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
                
      # Save the model checkpoint periodically.
      if step > 0 and (step % config.save_freq == 0 or (step + 1) == config.training_params.max_steps):
        checkpoint_path = os.path.join(config.ckpt_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
  config = config_interface.get_config(argv)
  train(config)

if __name__ == '__main__':
  tf.app.run()
