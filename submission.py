from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pandas as pd
import numpy as np
import tensorflow as tf
import PIL.Image

import config_interface
from input_manager import InputManager
from functools import reduce

def display(image):
  image = image - np.min(image)
  image = image * 255 / np.max(image)
  im = PIL.Image.fromarray(np.uint8(image))
  im.show()  

#%%
def evaluate(config):
  input_manager = InputManager(config)
  images, files = input_manager.submission_inputs() 
  with tf.variable_scope("inference"):    
    logits = config.inference(images, testing=True)

  # Restore the moving average version of the learned variables for eval.
  variable_averages = tf.train.ExponentialMovingAverage(config.training_params.moving_average_decay)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

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
      queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
      threads_per_qr = map(lambda qr: qr.create_threads(sess, coord=coord, daemon=True, start=True), queue_runners)
      threads = reduce(list.__add__, threads_per_qr)

      num_iter = math.ceil(config.dataset.submission_size / config.training_params.batch_size)
      
      step = 0
      preds = []
      while step < num_iter and not coord.should_stop():
        images_loc, logits_loc, labels_loc = sess.run([images, logits, files])
        display(images_loc[0])
        preds += zip(logits_loc, labels_loc)
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