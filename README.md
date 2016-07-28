General Purpose Image Recognition Framework
--------------------------------------------------------

The purpose of this framework is to provide a basis for image recognition, and go slightly further, in order to include complex data augmentations, while keeping the framework generic.

# Usage
The main files consist of files created ion order to train a network or to generate predictions based on an existing model. Those files, respectively named train.py and submission.py can be called with 6 arguments, which will provide the software with information on which dataset to work on, which model to do it with and how. The specifics of these arguments are provided in the README.me file found in the configurations/interfaces directory.

Those main file then make use of manager classes, such as update_manager or input_manager in order to manage the images provided to the network, the tensorflow session used as well as the updates needed for the training of the model.
