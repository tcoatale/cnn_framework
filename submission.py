from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import PIL.Image
from datetime import datetime
import pandas as pd

import config_interface
from input_manager import InputManager
from session_manager import SessionManager
from functools import reduce
import time

def display(image):
  image = image - np.min(image)
  image = image * 255 / np.max(image)
  im = PIL.Image.fromarray(np.uint8(image))
  im.show()  

#%%
class Evaluator:
  def __init__(self, config):
    self.config = config
    self.input_manager = InputManager(config)
    
  def run(self):
    with tf.Graph().as_default() as g:
      images, self.labels = self.input_manager.evaluation_inputs()  
      with tf.variable_scope("inference"):    
        self.logits = self.config.inference(images, testing=True) 
      self.loss = self.config.evaluation_loss(self.logits, self.labels)
    
      # Restore the moving average version of the learned variables for eval.
      variable_averages = tf.train.ExponentialMovingAverage(self.config.training_params.moving_average_decay)
      variables_to_restore = variable_averages.variables_to_restore()
      self.saver = tf.train.Saver(variables_to_restore)
    
      # Build the summary operation based on the TF collection of Summaries.
      self.summary_op = tf.merge_all_summaries()
      self.summary_writer = tf.train.SummaryWriter(self.config.eval_dir, g)

      while(True):    
        # Run evaluation
        try:
          self.evaluate_once()
        except:
          print('oHo')            
        time.sleep(self.config.training_params.eval_interval_secs)

  def evaluate_once(self):
    session_manager = SessionManager(self.config)
    global_step, sess = session_manager.restore(self.saver)  
    coord = tf.train.Coordinator()
    try:
      queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
      threads_per_qr = map(lambda qr: qr.create_threads(sess, coord=coord, daemon=True, start=True), queue_runners)
      threads = reduce(list.__add__, threads_per_qr)
      num_iter = math.ceil(self.config.dataset.valid_size / self.config.training_params.batch_size)
      
      step = 0
      losses = []
      while step < num_iter and not coord.should_stop():
        loss, logits, labels = sess.run([self.loss, self.logits, self.labels])
        losses += [loss]
        step += 1
        
      average_loss = np.mean(losses)
      format_str = ('%s: loss = %.8f')
      print (format_str % (datetime.now(), average_loss))
  
      summary = tf.Summary()
      summary.ParseFromString(sess.run(self.summary_op))
      summary.value.add(tag='evaluation_loss', simple_value=average_loss)
      self.summary_writer.add_summary(summary, global_step)    
      
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
    except Exception as e:
      coord.request_stop(e)
      
class SubmissionManager:
  def __init__(self, config):
    self.config = config
    self.input_manager = InputManager(config)
    
  def run(self):     
    images, files = self.input_manager.submission_inputs()  
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
  config = config_interface.get_config()  
  if argv and len(argv) == 2 and argv[1] =='s':
    print('\nSwitching to submission mode\n')
    submission_manager = SubmissionManager(config)
    submission_manager.run()
  else:
    evaluator = Evaluator(config)
    evaluator.run()

  

if __name__ == '__main__':
  tf.app.run()