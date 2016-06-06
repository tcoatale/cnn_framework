from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import matplotlib.pyplot as plt
import pylab
import time
import pickle
import numpy as np
import tensorflow as tf

import model
import application_interface
application = application_interface.get_application()

#%%
def submission():
  with tf.Graph().as_default() as g:
    images, labels = model.inputs('submission') 
    logits = model.inference(images)
    
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(application.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
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
        labels = sess.run([labels])
        logits = sess.run([logits])
        
        print(labels[0])
        print(logits[0])
        break

        
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
  submission()


if __name__ == '__main__':
  tf.app.run()