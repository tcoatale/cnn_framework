# -*- coding: utf-8 -*-
import os
import preprocessing


#.interfaces.dataset_interface as interface 

#%%
resize = 64
batch_size = 4000
train_ratio = 0.75

managers = interface.get_dataset_managers('driver')
dest_dir = os.path.join(managers.dest_dir_base, str(resize))

if not os.path.isdir(dest_dir):
  os.mkdir(dest_dir)

#%%
image_manager = managers.ImageManager((resize, resize), managers.aug_dir)
preprocessing_manager = managers.PreprocessingManager(data_dir=managers.data_dir,
                                             submission_dir=managers.submission_dir, 
                                             dest_dir=dest_dir, 
                                             train_ratio=train_ratio, 
                                             batch_size=batch_size)

preprocessing_manager.run(image_manager)