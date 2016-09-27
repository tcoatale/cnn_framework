# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from functools import reduce
import numpy as np
import glob 
import struct
import skimage.io

label_dict = {'GBM': 0, 'meningioma': 1}

class Dataset:
  def __init__(self, data):
    self.data = data
  
  def sparse_to_dense_id(self, file_id_tensor):
    identifier_bytes = self.data['bytes']['identifier']
    
    file_id_tensor_int32 = tf.cast(file_id_tensor, dtype=tf.int32)
    
    multiplier_np = 256**(np.linspace(identifier_bytes-1, 0, num=identifier_bytes))
    multiplier = tf.constant(multiplier_np)
    multiplier_int32 = tf.cast(multiplier, dtype=tf.int32)

    file_id = tf.reduce_sum(tf.mul(file_id_tensor_int32, multiplier_int32))
    return file_id
  
  def retrieve_file_id(self, file_number):
    return 'img_' + str(file_number) + '.jpg'
  
  def byte_form(input):
    return np.array(list(struct.unpack('4B', struct.pack('>I', int(input)))), dtype=np.uint8)

  
