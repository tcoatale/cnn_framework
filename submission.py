from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pandas as pd
import time
import PIL
import numpy as np
import tensorflow as tf

import model
import application_interface
application = application_interface.get_application()

#%%
def eval_once(saver, summary_writer, summary_op, images, logits, labels):
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
      while step < num_iter and not coord.should_stop():
        logits_loc, labels_loc, images = sess.run([logits, labels, images])
        #image = images[0]
        #image = image + np.min(image)
        #image = image * 255 / np.max(image)
        #im = PIL.Image.fromarray(np.uint8(image))
        #im.show()
        step += 1

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  with tf.Graph().as_default() as g:
    images, labels = model.inputs('submission') 
    _, logits = model.inference(images)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(application.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(application.eval_dir, g)

    eval_once(saver, summary_writer, summary_op, images, logits, labels)

def main(argv=None):
  evaluate()


if __name__ == '__main__':
  tf.app.run()