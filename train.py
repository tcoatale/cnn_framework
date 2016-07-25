from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import PIL.Image

from update_manager import UpdateManager
from input_manager import InputManager
import configurations.interfaces.configuration_interface as config_interface

def display(image):
  image = image - np.min(image)
  image = image * 255 / np.max(image)
  im = PIL.Image.fromarray(np.uint8(image))
  im.show()

def train(config):
  """Train model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    
    input_manager = InputManager(config)
    update_manager = UpdateManager(config)

    # Get images and labels for dataset.
    training_images, training_labels = input_manager.distorted_inputs()

    tf.image_summary('images', training_images, max_images=64)

    # Build a Graph that computes the logits predictions from the inference model.
    with tf.variable_scope("inference"):
      training_logits = tf.to_double(config.inference(training_images))
                        
    # Calculate loss.
    total_loss = update_manager.training_loss(training_logits, training_labels)
    classirate = config.loss.classirate(config.dataset, training_logits, training_labels)
   
    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = update_manager.update(total_loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
   
    summary_writer = tf.train.SummaryWriter(config.log_dir, sess.graph)

    for step in xrange(config.training_params.max_steps):
      start_time = time.time()
      _, _, _, total_loss_value, classirate_value = sess.run([training_logits, 
                                      training_labels, 
                                      train_op, 
                                      total_loss, 
                                      classirate])
                                      
      duration = time.time() - start_time

      assert not np.isnan(total_loss_value), 'Model diverged with loss = NaN'

      if step % config.display_freq == 0:
        num_examples_per_step = config.training_params.batch_size
        examples_per_sec = num_examples_per_step / duration
        print(datetime.now(), 'Step', step, 'Speed:', int(examples_per_sec), end='\t')
        print('Training loss:', total_loss_value, 'Classirate:', classirate_value, end='\n')
        
      if step % config.summary_freq == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
                
      # Save the model checkpoint periodically.
      if step > 0 and (step % config.save_freq == 0 or (step + 1) == config.training_params.max_steps):
        checkpoint_path = os.path.join(config.ckpt_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
  if argv and len(argv) == 7:
    dataset_name, dataset_size, training, loss_name, model_name, model_size = argv[1:] 
  else:
    dataset_name = 'driver'
    dataset_size = '32'
    training = 'fast'
    loss_name = 'driver'
    model_name = 'basic'
    model_size = 'normal'
    
  config = config_interface.get_config(dataset_name, dataset_size, training, loss_name, model_name, model_size)    
  train(config)


if __name__ == '__main__':
  tf.app.run()