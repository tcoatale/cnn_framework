# -*- coding: utf-8 -*-
import json

name_to_file_dict = {'standard': 'training_params/standard.json'}

def get_params(name):
  file_path=name_to_file_dict[name]
       
  with open(file_path) as data_file:  
    return json.load(data_file)
