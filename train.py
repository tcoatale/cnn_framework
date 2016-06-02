from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
import model
import application_interface
application = application_interface.get_application()

def train():
  """Train model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for dataset.
    images, labels = model.distorted_inputs()
    #tf.image_summary('images', images, max_images=128)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images)

    # Calculate loss.
    loss = model.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=application.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(application.log_dir, sess.graph)

    for step in xrange(application.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % application.display_freq == 0:
        num_examples_per_step = application.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.8f (%.1f examples/sec; %.3f sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

      if step % application.summary_freq == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        
      # Save the model checkpoint periodically.
      if step % application.save_freq == 0 or (step + 1) == application.max_steps:
        checkpoint_path = os.path.join(application.ckpt_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
  train()


if __name__ == '__main__':
  tf.app.run()