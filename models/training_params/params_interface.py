# -*- coding: utf-8 -*-
import json

name_to_file_dict = {'standard': 'standard.json'}

class ParamsInterface:
  def __init__(self, name):
    self.file_path=name_to_file_dict[name]

  def parse(self):        
    with open(self.file_path) as data_file:  
      return json.load(data_file)
