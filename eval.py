from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import model
import application_interface
application = application_interface.get_application()

#%%
def eval_once(saver, summary_writer, summary_op, images, logits, loss_function):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """    
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(application.ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

      num_iter = int(math.ceil(application.num_examples / application.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * application.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        images, loss = sess.run([images, loss_function])
        true_count += np.sum(loss)
        step += 1

      # Compute precision @ 1.  with tf.Session() as sess:

      precision = true_count / total_sample_count
      print('%s: evaluation loss: %.8f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  with tf.Graph().as_default() as g:
    images, labels = model.inputs('eval') 
    logits = model.inference(images)
    loss_function = application.evaluation_loss(logits, labels)
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(application.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(application.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, summary_op, images, logits, loss_function)
      if application.run_once:
        break
      time.sleep(application.eval_interval_secs)

def main(argv=None):
  evaluate()


if __name__ == '__main__':
  tf.app.run()