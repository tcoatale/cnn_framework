# * coding: utf8 *
from preprocessing.interfaces.dataset_interface import get_dataset
from preprocessing.managers import preprocessing_manager

resize = [64, 64]
batch_size = 4000

dataset = get_dataset('pn')
preprocessing_manager = preprocessing_manager.PreprocessingManager(dataset=dataset, 
                                                                   resize=resize,
                                                                   batch_size=batch_size)
preprocessing_manager.run()
