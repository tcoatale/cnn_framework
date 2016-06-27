from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import PIL.Image

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
      step = 0
      losses = []
      while step < num_iter and not coord.should_stop():
        preds, loss = sess.run([logits, loss_function])
        #image = images[0]
        #image = image + np.min(image)
        #image = image * 255 / np.max(image)
        #im = PIL.Image.fromarray(np.uint8(image))
        #im.show()
        losses += [loss]
        step += 1

      # Compute precision @ 1.  with tf.Session() as sess:
      average_loss = np.mean(losses)
      print('%s: evaluation loss: %.8f' % (datetime.now(), average_loss))
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  with tf.Graph().as_default() as g:
    images, labels = model.inputs('eval') 
    logits = model.inference(images)
    loss_function = application.loss(logits, labels)
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