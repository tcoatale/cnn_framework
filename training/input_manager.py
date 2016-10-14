from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import glob
import numpy as np

class FileManager:
  def __init__(self, model):
    self.model = model
    
  def get_files(self, type):
    filenames = glob.glob(os.path.join(self.model.dataset.data['directories']['processed'], type + '*'))    
    if filenames == []:
      raise ValueError('No such batch files')
      
    filename_queue = tf.train.string_input_producer(filenames)
    return filename_queue

class InputManager:
  def __init__(self, model):
    self.model = model

  def read_binary(self, filename_queue):
    image_dim = self.model.dataset.data['images']['resized']
    height, width, depth = image_dim['height'], image_dim['width'], image_dim['depth']
    additional_channels = self.model.dataset.get_number('channel')
    
    identifier_bytes = self.model.dataset.data['bytes']['identifier']
    label_bytes = self.model.dataset.data['bytes']['label']
    image_bytes = width * height * depth
    aug_filter_bytes = width * height * additional_channels
    aug_feature_bytes = self.model.dataset.get_number('features')
    
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
    
    
    # Cast the label to float
    label = tf.cast(label_read, tf.float32)
    
    # Fetch the image by converting from [depth * height * width] to [width, height, depth].
    image = tf.reshape(image_read, [depth, height, width])
    transposed_image = tf.cast(tf.transpose(image, [2, 1, 0]), tf.float32)
    
    add_filter = tf.reshape(add_filter_read, [additional_channels, height, width])
    transposed_add_filter = tf.cast(tf.transpose(add_filter, [2, 1, 0]), tf.float32)
    
    add_features = tf.cast(add_features_read, tf.float32)  
    return id_read, label, transposed_image, transposed_add_filter, add_features
  
  
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
    min_queue_examples = int(self.model.dataset.data['set_sizes'][type] * 0.8)
    print ('Filling queue with %d images before starting to train. This will take a few minutes.' % min_queue_examples)

    # Get file queue
    file_manager = FileManager(self.model)
    filename_queue = file_manager.get_files(type)

    # Read examples from files in the filename queue.
    id, label, image, add_filter, add_features = self.read_binary(filename_queue)

    # Distort image
    processed_image, processed_filter = self.model.dataset.process_inputs(image, add_filter, distort=distorted)
    
    # Generate a batch of images and labels by building up a queue of examples.
    return self._generate_image_and_label_batch(id=id, 
                                                label=label, 
                                                image=processed_image, 
                                                add_filter=processed_filter, 
                                                add_features=add_features,
                                                min_queue_examples=min_queue_examples, 
                                                batch_size=self.model.params['batch_size'], 
                                                shuffle=shuffle)