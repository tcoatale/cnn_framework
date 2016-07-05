from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pandas as pd
import numpy as np
import tensorflow as tf

import config_interface
from input_manager import InputManager

#%%
def evaluate(config):
  input_manager = InputManager(config)
  images, files = input_manager.submission_inputs() 
  with tf.variable_scope("inference") as scope:    
    logits = config.inference(images, testing=True)

  # Restore the moving average version of the learned variables for eval.
  #variable_averages = tf.train.ExponentialMovingAverage(config.training_params.moving_average_decay)
  #variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver()

  # Build the summary operation based on the TF collection of Summaries.
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(config.ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
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

      num_iter = int(math.ceil(config.dataset.submission_size/ config.training_params.batch_size))
      
      step = 0
      while step < num_iter and not coord.should_stop():
        logits_loc, labels_loc = sess.run([logits, files])
        print(logits_loc[0])
        step += 1

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
def main(argv=None):
  config = config_interface.get_config()
  evaluate(config)


if __name__ == '__main__':
  tf.app.run()