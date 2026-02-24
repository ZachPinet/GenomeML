This is a file-by-file breakdown of the structure of the GenomeML project.
It aims to document what each file does and how it interacts with other files.

# Inputs

There are two inputs that GenomeML requires. The first is the inputs/columns/ subfolder, which must contain 'column' files. Each file must have n lines and n float values, with each value on its own line, being separated only by the newlines. The names of each column file can be anything.

# SRC

## main.py
This file is the entry point for the program. It updates the configs, then determines whether to start the program with or without the GUI.

## config_local.txt
The name 'config_local.txt' is hardcoded in run_window.py. It is first created and populated by run_window.py and stores local config values that will override the default config values. It can be directly edited by the user, and the changes will be applied.

## config.py
This file establishes a dictionary of default config values. It then loads and parses the values from config_local.txt, uses them to replace their respective default values, and exports them as module-level variables.
These module-level variables are then called by other files after importing config.py.
The code to update the module-level variables is executed as soon as the program starts, and after each time the Run Model button is pressed in the Run Window.

## model.py
This file contains the model architecture to build a model when a workflow file calls for it. 

### Batch Correction Library
When the USE_BATCH_CORRECTION config is True, the model architecture will include a Batch Correction Layer, defined by the BatchCorrectionLayer library. This library is based off of tf.keras.layers.Layer and includes functions called by Keras.

# GUI

## Run Window

### Run Model Button
When this button is clicked, the chosen configs are saved to the config_local.txt and config.py is reloaded so that its module-level variables can be updated with the new local configs.

# Workflows
