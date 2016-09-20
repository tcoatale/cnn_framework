# -*- coding: utf-8 -*-
import json

class JsonInterface:
  def __init__(self, file_path):
    self.file_path=file_path

  def parse(self):        
    with open(self.file_path) as data_file:  
      return json.load(data_file)