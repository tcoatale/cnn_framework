from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pandas as pd
import time
import PIL
import numpy as np
import tensorflow as tf
import os

import config_interface
from input_manager import InputManager


base_dir = 'ckpt/driver_augmented'
ckpt_dir = os.path.join(base_dir, '2016-7-2-18-38-26')

#%%
#def eval_once(saver, summary_writer, summary_op, images, logits, labels):
  


def evaluate(config):
  with tf.Graph().as_default() as g:
    
    input_manager = InputManager(config)
    submission_images, submission_files = input_manager.submission_inputs()
    submission_logits = config.inference(submission_images, testing=True)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(config.training_params.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    tf.image_summary('images', submission_images, max_images=64)

    
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(config.eval_dir, g)
    
    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(ckpt_dir)
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
  
        num_iter = int(math.ceil(config.dataset.submission_size / config.dataset.batch_size)) + 1
        
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, 0)
        step = 0

        while step < num_iter and not coord.should_stop():          
          logits, files, images = sess.run([submission_logits, submission_files, submission_images])
          image = images[0]
          image = image + np.min(image)
          image = image * 255 / np.max(image)
          im = PIL.Image.fromarray(np.uint8(image))
          im.show()
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