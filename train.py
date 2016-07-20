from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import PIL.Image

from update_manager import UpdateManager
from input_manager import InputManager
import config_interface

def display(image):
  image = image - np.min(image)
  image = image * 255 / np.max(image)
  im = PIL.Image.fromarray(np.uint8(image))
  im.show()

def train(config):
  """Train model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    
    input_manager = InputManager(config)
    update_manager = UpdateManager(config)

    # Get images and labels for dataset.
    training_images, training_labels = input_manager.distorted_inputs()

    tf.image_summary('images', training_images, max_images=64)

    # Build a Graph that computes the logits predictions from the inference model.
    with tf.variable_scope("inference"):
      training_logits = tf.to_double(config.inference(training_images))
                        
    # Calculate loss.
    loss = update_manager.training_loss(training_logits, training_labels)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = update_manager.update(loss, global_step)

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
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % config.display_freq == 0:
        num_examples_per_step = config.training_params.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration) 
        format_str = ('%s: step %d, loss = %.8f (%.1f examples/sec; %.3f sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

      if step % config.summary_freq == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
                
      # Save the model checkpoint periodically.
      if step > 0 and (step % config.save_freq == 0 or (step + 1) == config.training_params.max_steps):
        checkpoint_path = os.path.join(config.ckpt_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
  if argv and len(argv) == 3:
    dataset, model = argv[1:] 
  else:
    dataset, model = 'pn', 'normal_alex'
  config = config_interface.get_config(dataset=dataset, model=model)    
  train(config)


if __name__ == '__main__':
  tf.app.run()