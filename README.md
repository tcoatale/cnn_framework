General Purpose Image Recognition Framework
--------------------------------------------------------

The purpose of this framework is to provide a basis for image recognition, and go slightly further, in order to include complex data augmentations, while keeping the framework generic.

# Applications
This framework consists of three different applications, united into one global project.

- The first application consists of using raw images in order to extract features and store them in a local directory.
- The second application uses local files and images in order to create training and testing batches in a general format.
- The third application uses those local files together with training configurations in order to train and evaluate networks.

## Extraction
In order to extract the features, the extract.py file calls files from the preprocessing/datasets/ directory. It is in this directory that all datasets should be registered, providing the appropriate methods.

## Preprocessing
The preprocessing application, called from the preprocess.py file, then uses the same dataset file and an Image manager in order, for each image, to gather all of the information and to write it in a single numerical line. Those lines are then passed through batches and written locally.

## Training and Evaluation
In order to train a network, the train.py, retrain.py and evaluation.py files take 6 arguments, whose default values are set in the configurations/interfaces/configuration_interface.py file. Those 6 arguments are explained in the configuratoins/interfaces directory.
