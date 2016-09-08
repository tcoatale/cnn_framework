General Purpose Image Recognition Framework
--------------------------------------------------------

The purpose of this framework is to provide a basis for image recognition, and go slightly further, in order to include complex data augmentations, while keeping the framework generic.

# Usage
The main files consist of files created in order to train a network and evaluate it. Those files, respectively named train.py and submission.py can be called with 6 arguments, which will provide the software with information on which dataset to work on, which model to do it with and how. The specifics of these arguments are provided in the README.me file found in the configurations/interfaces directory. the retrain.py file with the correct arguments will restart the training of a network based on the last stored version of the model

Those main file then make use of manager classes, such as update_manager or input_manager in order to manage the images provided to the network, the tensorflow session used as well as the updates needed for the training of the model.
