import datasets.driver_augmented as dataset
import trainings.driver_augmented as training

#%%
dataset.classes
training.log_dir

#%%
application = 'driver_augmented'

#%%
import datasets.application as dataset

#%%
#%%
import trainings.application.__str__ as training

#%%
import training
application_training = getattr(training, application)()

#%%
application_training.batch_size
#%%
application_training