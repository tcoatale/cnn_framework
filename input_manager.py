from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import glob
import numpy as np

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
    width, height, depth = self.config.dataset.original_shape

    identifier_bytes = self.config.dataset.identifier_bytes
    label_bytes = self.config.dataset.label_bytes
    image_bytes = width * height * depth
    aug_filter_bytes = width * height * self.config.dataset.additional_filters
    aug_feature_bytes = self.config.dataset.aug_feature_bytes
    
    segments_lengths = [identifier_bytes, label_bytes, image_bytes, aug_filter_bytes, aug_feature_bytes]
    segments_starts = np.cumsum([0] + segments_lengths[:-1])
    record_bytes = np.sum(segments_lengths)

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # Get the appropriate bytes depending on the positions within the vector
    id_read = tf.slice(record_bytes, [segments_starts[0]], [segments_lengths[0]], name='id_slicer')
    label_read = tf.slice(record_bytes, [segments_starts[1]], [segments_lengths[1]], name='label_slicer')
    image_read = tf.slice(record_bytes, [segments_starts[2]], [segments_lengths[2]], name='image_slicer')
    add_filter_read = tf.slice(record_bytes, [segments_starts[3]], [segments_lengths[3]], name='filter_slicer')
    add_features_read = tf.slice(record_bytes, [segments_starts[4]], [segments_lengths[4]], name='feature_slicer')
    
    # Fetch the actual id of the file based on the bytes read    
    id = self.config.dataset.sparse_to_dense_id(id_read)
    
    # Cast the label to float
    label = tf.cast(label_read, tf.float32)
    
    # Fetch the image by converting from [depth * height * width] to [width, height, depth].
    image = tf.reshape(image_read, [depth, height, width])
    transposed_image = tf.cast(tf.transpose(image, [2, 1, 0]), tf.float32)
    
    add_filter = tf.reshape(add_filter_read, [1, height, width])
    transposed_add_filter = tf.cast(tf.transpose(add_filter, [2, 1, 0]), tf.float32)
    
    add_features = tf.cast(add_features_read, tf.float32)  
    return id, label, transposed_image, transposed_add_filter, add_features
  
  
  def _generate_image_and_label_batch(self, id, label, image, add_filter, add_features, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    
    if shuffle:
      ids, labels, images, add_filters, features = tf.train.shuffle_batch(
          [id, label, image, add_filter, add_features],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
    else:
      ids, labels, images, add_filters, features = tf.train.batch(
          [id, label, image, add_filter, add_features],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size)
  
    return ids, labels, images, add_filters, features
      
  def get_inputs(self, type='train', distorted = True, shuffle = True):
    min_queue_examples = int(self.config.dataset.train_size * 0.4)
    print ('Filling queue with %d images before starting to train. This will take a few minutes.' % min_queue_examples)

    # Get file queue
    file_manager = FileManager(self.config)
    filename_queue = file_manager.get_files(type)

    # Read examples from files in the filename queue.
    id, label, image, add_filter, add_features = self.read_binary(filename_queue)

    # Distort image
    processed_image, processed_filter = self.config.dataset.process_inputs(image, add_filter, distort=distorted)
    
    width, height, depth = self.config.dataset.imshape
    filter_num = self.config.dataset.additional_filters
    depth -= filter_num
    
    # Generate a batch of images and labels by building up a queue of examples.
    return self._generate_image_and_label_batch(id=id, 
                                                label=label, 
                                                image=processed_image, 
                                                add_filter=processed_filter, 
                                                add_features=add_features,
                                                min_queue_examples=min_queue_examples, 
                                                batch_size=self.config.training_params.batch_size, 
                                                shuffle=shuffle)