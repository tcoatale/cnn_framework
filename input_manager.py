from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import glob

class FileManager:
  def __init__(self, config):
    self.config = config
    
  def get_files(self, type):
    filenames = glob.glob(os.path.join(self.config.dataset.data_dir, type + '*'))    
    if filenames == []:
      raise ValueError('No such batch files')
      
    filename_queue = tf.train.string_input_producer(filenames)
    return filename_queue

class InputManager:
  def __init__(self, config):
    self.config = config

  def read_binary(self, filename_queue):  
    identifier_bytes = self.config.dataset.identifier_bytes
    label_bytes = self.config.dataset.label_bytes
    image_bytes = self.config.dataset.image_bytes

    record_bytes = identifier_bytes + label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # Get the appropriate bytes depending on the positions within the vector
    id_read = tf.slice(record_bytes, [0], [identifier_bytes])
    image_read = tf.slice(record_bytes, [identifier_bytes], [image_bytes])
    label_read = tf.slice(record_bytes, [image_bytes + identifier_bytes], [label_bytes])
    
    # Fetch the actual id of the file based on the bytes read    
    id = self.config.dataset.sparse_to_dense_id(id_read)
    
    # Cast the label to float
    label = tf.cast(label_read, tf.float32)
    
    # Fetch the image by converting from [depth * height * width] to [height, width, depth].
    height, width, depth = self.config.dataset.original_shape
    image = tf.reshape(image_read, [depth, height, width])
    transposed_image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
  
    return id, label, transposed_image
  
  
  def _generate_image_and_label_batch(self, id, label, image, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    
    if shuffle:
      ids, labels, images = tf.train.shuffle_batch(
          [id, label, image],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
    else:
      ids, labels, images = tf.train.batch(
          [id, label, image],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size)
  
    return ids, labels, images
      
  def get_inputs(self, type='train', distorted = True, shuffle = True):
    min_queue_examples = int(self.config.dataset.train_size * 0.4)
    print ('Filling queue with %d images before starting to train. This will take a few minutes.' % min_queue_examples)

    # Get file queue
    file_manager = FileManager(self.config)
    filename_queue = file_manager.get_files(type)

    # Read examples from files in the filename queue.
    id, label, image = self.read_binary(filename_queue)

    # Distort image
    processed_image = self.config.dataset.process_inputs(image, distort=distorted)
    whitened_image = tf.image.per_image_whitening(processed_image)

    # Generate a batch of images and labels by building up a queue of examples.
    return self._generate_image_and_label_batch(id=id, 
                                                label=label, 
                                                image=whitened_image, 
                                                min_queue_examples=min_queue_examples, 
                                                batch_size=self.config.training_params.batch_size, 
                                                shuffle=shuffle)