This is a brief description of each config option.

### SINGLE_FILE_NUM
This is the column file number that is used when the File Selection option is set to Single File instead of Range of Files.

### RANGE_START_FILE_NUM and RANGE_END_FILE_NUM
These are the the column file numbers that are the first and last to be used when the File Selection option is set to Range of Files instead of Single File.

### DO_RANGE
When this is set to False, only the column file specified with SINGLE_FILE_NUM will be used. When this is set to True, every column file in the range RANGE_START_FILE_NUM to RANGE_END_FILE_NUM will be used, each in their own run, one after the other.

### DO_DOUBLE_COLUMNS
When this is True, the Double Columns workflow mode will be selected. The model will be trained on DOUBLE_TRAIN_FILE and will be tested on the specified Single File or Range of Files.

### DO_ENSEMBLE
When this is True, the Ensemble workflow mode will be selected. This will run multiple models and average their predictions. 

### DO_PCA
When this is True, the Principle Components Analysis (PCA) workflow mode will be selected. This takes all columns and creates clustered components from them. A model is trained on each component.

### DO_SINGLE_COLUMN
When this is True, the Single Column workflow mode will be selected. The model will be trained and tested on different parts of the same column specified by Single File or Range of Files.

### KFOLD
This is an option for Single Column mode. When True, K-fold cross-validation is used. Data from each fold is stored and used in a combined graph once all folds have run.

### PCA_COMPONENTS
This is a variable for PCA mode that determines how many components are created.

### DATA_SPLITS
This is a variable for Ensemble mode that determines how many models are run and averaged together.

### DOUBLE_TRAIN_FILE
This is the column file number that Double Columns mode trains on.

### MAX_SEQS
This is the maximum number of sequence-value pairs that the model will consider when splitting train/test data.

### TRAIN_PERCENTAGE
This is an integer representing the percent (%) of a column file that is used for model training. Parts of a file that are unused during training will be used during testing. If the Ensemble mode is active, the TRAIN_PERCENTAGE must be a multiple of (100 / DATA_SPLITS).

### VERBOSE
This is the verbosity of the model output. It is an integer; either 0, 1, or 2. It affects how the model displays its progress in the terminal.

### MAKE_PLOT
When this is True, a graph will automatically be created at the end of a workflow from the data generated. Currently the only graph created is the heatmap scatterplot.

### SHOW_BOUNDS
When this is True, the generated heatmap scatterplot will display a LOESS curve as well as upper and lower standard deviation bounds.

### STD_MULTIPLIER
This is the value for the standard deviation used to create upper and lower bounds. These are displayed on the heatmap scatterplot. Any predicted values outside these bounds are classified as outliers.

### FRAC
This is a value between 0-1 that affects the smoothness and sensitivity of the LOESS curve.

### OUTLIER_MODE
This is a string that can be 'simple', 'complex', 'both', or 'off'. If the Outlier Mode is 'simple' or 'both', a file will be created with the outlier data from a single run in outputs/outliers_simple/. If the mode is 'complex' or 'both', a file will be created or edited with the outlier data from the most recent run in outputs/outliers_complex/. Any mode besides 'off' will also create or edit a file with the percentage of outliers from the most recent run in outputs/outliers_percent/. If the mode is 'off', no outliers will be logged at all.

### MODEL
This is an integer that corresponds to one of the files in the src/models/ folder. The selected file will have its build_model() function used. For example, if MODEL=1, model1_basic.py will be the model used.

### RANDOM_SEED
This seed is used for all randomness so that it may reproduceable.

### WINDOW
When this is true, the program will start by opening up the interactive window. When false, the window will never open, and the program will run without interaction using the saved config values.

### USE_BATCH_CORRECTION
When this is true, the model will use batch correction based on batches derived from the names of each column file.

### COMPLEX_OUTLIER_FILE
This is the string name for the .txt file that will receive outlier data from a run if the Outlier Mode is 'complex' or 'both'. The file with this name will be placed into outputs/outliers_complex/. If the file already exists and has data, new data will be appended to it, rather than overwriting.

### PERCENT_OUTLIER_FILE
This is the string name for the .txt file that will receive outlier data from a run if outliers are enabled. The file with this name will be placed into outputs/outliers_percent/. If the file already exists and has data, new data will be appended to it, rather than overwriting.

### RUN_DATA_FILE
This is the string name for the .txt file that will receive data from a run. The file with this name will be placed into outputs/run_data/. If the file already exists and has data, new data will be appended to it, rather than overwriting.