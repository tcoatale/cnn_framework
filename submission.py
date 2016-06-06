from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pandas as pd
import numpy as np
import tensorflow as tf

import model
import application_interface
application = application_interface.get_application()

#%%
def submission():
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
  print('Checkpoint restored.')

  num_iter = int(math.ceil(application.num_examples / application.batch_size))
  step = 0
  labels_loc = []
  logits_loc = []
  
  while step < num_iter:
    labels_loc += sess.run([labels])
    logits_loc += sess.run([logits])
    step += 1
        
  df = pd.DataFrame(np.vstack([labels_loc, logits_loc]))
  df.to_csv('file.csv')

def main(argv=None):
  submission()


if __name__ == '__main__':
  tf.app.run()