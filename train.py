from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
from time import gmtime, strftime

from training.update_manager import UpdateManager
from training.input_manager import InputManager
from training.session_manager import SessionManager

from models.model import Model

def train(model):
  """Train model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    input_manager = InputManager(model)
    update_manager = UpdateManager(model)

    # Get images and labels for dataset.
    with tf.variable_scope("training_inputs") as scope:
      training_ids, training_labels, training_images, training_add_filters, training_features = input_manager.get_inputs()
    with tf.variable_scope("eval_inputs") as scope:
      eval_ids,  eval_labels,  eval_images,  eval_add_filters,  eval_features = input_manager.get_inputs(type='test', distorted = False, shuffle = True)
    
    tf.image_summary('images', training_images, max_images=64)

    # Build a Graph that computes the logits predictions from the inference model.
    with tf.variable_scope("inference") as scope:
      training_logits = model.inference(training_images, training_add_filters, training_features)
      scope.reuse_variables()
      eval_logits = model.inference(eval_images, eval_add_filters, eval_features, testing=True)

    # Calculate loss.
    loss_training = model.loss(training_logits, training_labels)
    loss_eval = model.loss(eval_logits, eval_labels)
    tf.scalar_summary('loss_eval', loss_eval)

    total_loss = update_manager.training_loss(loss_training)

    classirate_training = model.classirate(training_logits, training_labels)
    classirate_eval = model.classirate(eval_logits, eval_labels)    

    tf.scalar_summary('classirate_training', classirate_training)
    tf.scalar_summary('classirate_eval', classirate_eval)
    
   
    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = update_manager.update(total_loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()


    if model.params['continue']:
      session_manager = SessionManager(model)
      current_step, sess = session_manager.restore(saver)
      
    else:   
      init = tf.initialize_all_variables()
      sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
      sess.run(init)
      current_step = 0

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.train.SummaryWriter(model.log_dir, sess.graph)

    for step in xrange(int(current_step), model.params['max_steps']):
      step_increment = global_step.assign(step)
      start_time = time.time()

      labels, logits = sess.run([training_labels, training_logits])
      train_loss_value, eval_loss_value = sess.run([loss_training, loss_eval])
      total_loss_value = sess.run([total_loss])[0]      
      sess.run([step_increment, train_op])
      duration = time.time() - start_time

      assert not np.isnan(total_loss_value), 'Model diverged with loss = NaN'

      if step % model.params['display_freq'] == 0:
        num_examples_per_step = model.params['batch_size']
        examples_per_sec = num_examples_per_step / duration

        print(strftime("%D %H:%M:%S", gmtime()), end=' ')
        print('Step', '%06d' % step, end=' ')
        print('Speed:', "%04d" % int(examples_per_sec), end=' ')
        print('Training loss:', total_loss_value, end=' ')
        print('T:', train_loss_value, end=' ')
        print('E:', eval_loss_value, end='\n')
        
      if step % model.params['summary_freq'] == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
                
      # Save the model checkpoint periodically.
      if step > 0 and (step % model.params['save_freq'] == 0 or (step + 1) == model.params['max_steps']):
        checkpoint_path = os.path.join(model.ckpt_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
  tf.set_random_seed(1)
  model = Model()
  train(model)

if __name__ == '__main__':
  tf.app.run()
