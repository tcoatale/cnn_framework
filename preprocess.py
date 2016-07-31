# * coding: utf8 *
from preprocessing.interfaces.dataset_interface import get_dataset_managers
from preprocessing.managers import preprocessing_manager

resize = 64
batch_size = 4000
train_ratio = 0.75

dataset_manager = get_dataset_managers('driver')
preprocessing_manager = preprocessing_manager.PreprocessingManager(dataset_manager=dataset_manager, 
                                                                   resize=resize,
                                                                   train_ratio=train_ratio, 
                                                                   batch_size=batch_size)
preprocessing_manager.run()