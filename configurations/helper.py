import os

def try_mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def initialize_directory(function, dataset_name, config_name):
  function_dataset_dir = os.path.join(function, dataset_name)
  specific_dir = os.path.join(function_dataset_dir, config_name)
  list(map(try_mkdir, [function_dataset_dir, specific_dir]))
  
def initialize_directories(log_dir, ckpt_dir, dataset_name, config_name):
  list(map(lambda d: initialize_directory(d, dataset_name, config_name), [log_dir, ckpt_dir]))