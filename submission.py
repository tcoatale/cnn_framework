from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import pandas as pd

import configurations.interfaces.configuration_interface as config_interface

from input_manager import InputManager
from session_manager import SessionManager
from functools import reduce
      
class SubmissionManager:
  def __init__(self, config):
    self.config = config
    self.input_manager = InputManager(config)
    
  def run(self):
      files, labels, images = input_manager.get_inputs(type='submission', distorted = False, shuffle = False)

    with tf.variable_scope("inference"):    
      logits = self.config.inference(images, testing=True)
        
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(self.config.training_params.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    self.saver = tf.train.Saver(variables_to_restore)
    
    session_manager = SessionManager(self.config)
    global_step, sess = session_manager.restore(self.saver)  
    coord = tf.train.Coordinator()
    try:
      queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
      threads_per_qr = map(lambda qr: qr.create_threads(sess, coord=coord, daemon=True, start=True), queue_runners)
      threads = reduce(list.__add__, threads_per_qr)
      num_iter = math.ceil(self.config.dataset.submission_size / self.config.training_params.batch_size)
      
      step = 0
      predictions_ids = []
      predictions = []
      while step < num_iter and not coord.should_stop():
        submission_files, submission_logits = sess.run([files, logits])
        submission_files = list(map(self.config.dataset.retrieve_file_id, submission_files))
        predictions += [submission_logits]
        predictions_ids += submission_files
        step += 1
        
      predictions = np.vstack(predictions)
      predictions = np.float32(predictions)
      predictions += 1e-3
      row_sums = np.reshape(predictions.sum(axis=1), [-1, 1])
      predictions /= row_sums
      
      df = pd.DataFrame(data=predictions)
      cols = list(map(lambda c: 'c' + str(c), range(10)))
      
      df.columns = cols
      df['img'] = predictions_ids
      
      df = df[['img'] + cols]
      df = df.drop_duplicates(subset = ['img'], keep='first')
      df.to_csv('submission.csv', index=False)
      
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
    except Exception as e:
      coord.request_stop(e)
      
def main(argv=None):
  config = config_interface.get_config(argv)
  submission_manager = SubmissionManager(config)
  submission_manager.run()

if __name__ == '__main__':
  tf.app.run()