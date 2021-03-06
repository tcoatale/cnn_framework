# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf

import configurations.interfaces.configuration_interface as config_interface
from input_manager import InputManager
from session_manager import SessionManager
from functools import reduce

#%%
class Evaluator:
  def __init__(self, config):
    self.config = config
    self.input_manager = InputManager(config)
    
  def run(self):
    with tf.Graph().as_default() as g:
      tf.set_random_seed(1)
      with tf.variable_scope("eval_inputs"):
        eval_ids,  eval_labels,  eval_images,  eval_add_filters,  eval_features = self.input_manager.get_inputs(type='test', distorted = False, shuffle = False)
      with tf.variable_scope("inference"):
        eval_logits = self.config.inference(eval_images, eval_add_filters, eval_features, testing=True)
        
      self.logits = eval_logits
      self.labels = eval_labels
      
      # Restore the moving average version of the learned variables for eval.
      variable_averages = tf.train.ExponentialMovingAverage(self.config.training_params.moving_average_decay)
      variables_to_restore = variable_averages.variables_to_restore()
      self.saver = tf.train.Saver(variables_to_restore)
   
      try:
        self.evaluate_once()
      except Exception as err:
        print('oHo: {0}'.format(err))            

  def evaluate_once(self):
    session_manager = SessionManager(self.config)
    global_step, sess = session_manager.restore(self.saver)  
    coord = tf.train.Coordinator()
    try:
      queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
      threads_per_qr = map(lambda qr: qr.create_threads(sess, coord=coord, daemon=True, start=True), queue_runners)
      threads = reduce(list.__add__, threads_per_qr)
      num_iter = math.ceil(self.config.dataset.set_sizes['test'] / self.config.training_params.batch_size)
      
      step = 0
      acc = []
      
      full_logits = []
      full_labels = []


      while step < num_iter and not coord.should_stop():
        logits, labels = sess.run([self.logits, self.labels])
        full_logits += [logits]
        full_labels += [labels]
        step += 1
      
      logits = np.vstack(full_logits)
      labels = np.vstack(full_labels)
      labels = np.array(labels, np.uint8).reshape((-1))
      
      good_logits = logits[np.arange(logits.shape[0]), labels]
      classirate = good_logits.sum() / logits.shape[0]
      
      print ('Classification rate:',  classirate)        
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
    except Exception as e:
      coord.request_stop(e)
      
def main(argv=None):
  config = config_interface.get_config(argv)
  evaluator = Evaluator(config)
  evaluator.run()

if __name__ == '__main__':
  tf.app.run()