from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce

import os
import tensorflow as tf
import glob

import config_interface
config = config_interface.get_config()

def read_binary(filename_queue, label_size_exception=False):
  """Reads and parses examples from binary data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class Record(object):
    pass
  result = Record()

  label_bytes = config.dataset.label_bytes if not label_size_exception else label_size_exception     
  result.height, result.width, result.depth = config.dataset.original_shape
  image_bytes = reduce(int.__mul__, config.dataset.original_shape)
  record_bytes = label_bytes + image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  label_array = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
  multiplier = list(map(lambda l: l * 2 ** 8, range(label_bytes)))
  
  result.label = tf.reduce_sum(tf.mul(label_array, multiplier))

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [result.depth, result.height, result.width])
  
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs():
  """Construct distorted input for training using the Reader ops.
  Args:

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = glob.glob(os.path.join(config.dataset.data_dir, 'data_batch_*'))
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_binary(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  float_image = config.dataset.distort_inputs(reshaped_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.8
  min_queue_examples = int(config.dataset.train_size * min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, config.training_params.batch_size, shuffle=True)

def training_inputs():
  label_size_exception = False
  filenames = glob.glob(os.path.join(config.dataset.data_dir, 'data_batch_*'))
  num_examples_per_epoch = config.dataset.train_size  

  return inputs(filenames, num_examples_per_epoch, label_size_exception)
  
def evaluation_inputs():
  label_size_exception = False
  filenames = glob.glob(os.path.join(config.dataset.data_dir, 'test_batch*'))
  num_examples_per_epoch = config.dataset.valid_size
  
  return inputs(filenames, num_examples_per_epoch, label_size_exception)
  
def submission_inputs():
  filenames = glob.glob(os.path.join(config.dataset.data_dir, 'submission_batch*'))
  num_examples_per_epoch = config.dataset.valid_size
  label_size_exception = config.dataset.id_bytes    

  return inputs(filenames, num_examples_per_epoch, label_size_exception)

def inputs(filenames, num_examples_per_epoch, label_size_exception):
  """Construct input for evaluation using the Reader ops.
  Args:
    num_examples_per_epoch: number of images per epoch
    filenames: the corresponding files
    batch_size: Number of images per batch.
    label_size_exception: Number to check in the case of submission files
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_binary(filename_queue, label_size_exception)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = config.dataset.imsize
  width = config.dataset.imsize

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_queue_examples = int(num_examples_per_epoch)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, config.training_params.batch_size, shuffle=False)