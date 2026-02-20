import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from src import config
from src.data_loading import load_data, load_all_columns
from src.workflows.double_columns import double_columns
from src.workflows.ensemble import ensemble
from src.workflows.pca import pca_values
from src.workflows.single_columns import single_column


# Run the appropriate training workflow based on current config values.
def run_training():
    # Make sure any randomization is repeatable.
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    # Access the files in the 'inputs' folder.
    cwd = Path.cwd()
    columns_dir = cwd / 'inputs' / 'columns'
    seq_file = cwd / 'inputs' / 'seqs.fa'
    max_seqs = config.MAX_SEQS
        
    # Access files for detect_outliers() function.
    outputs_dir = cwd / 'outputs'
    outputs_dir.mkdir(exist_ok=True)  # Ensure outputs directory exists
    output_file = outputs_dir / 'outliers.txt'
    pct_file = outputs_dir / 'outlier_percents.txt'

    # Train and test on the same column's values only.
    if config.DO_SINGLE_COLUMN:
        print("Single column mode enabled.")

        # Select the values file.
        for i in range(
            config.RANGE_START_FILE_NUM, config.RANGE_END_FILE_NUM + 1
        ):
            if config.DO_RANGE == False:
                i = config.SINGLE_FILE_NUM
            
            # Reset seeds for identical model behavior each iteration.
            random.seed(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)
            tf.random.set_seed(config.RANDOM_SEED)

            values_file = sorted(columns_dir.glob("*.txt"))[i - 1]
            x, y, batch = load_data(seq_file, values_file, max_seqs)
            col_name = values_file.name[:-4]

            # Pass the variables to single_columns workflow.
            single_column(
                x, y, batch, 
                kfold=config.KFOLD, 
                col_name=col_name, 
                show_bounds=config.SHOW_BOUNDS, 
                std_multiplier=config.STD_MULTIPLIER, 
                frac=config.FRAC, 
                output_file=output_file, 
                pct_file=pct_file, 
                mode=config.OUTLIER_MODE
            )

            if config.DO_RANGE == False:
                break

    # Use Parts Cluster Analysis (PCA) on all columns.
    elif config.DO_PCA:
        print("PCA mode enabled - loading full dataset.")

        for i in range(
            config.RANGE_START_FILE_NUM, config.RANGE_END_FILE_NUM + 1
        ):
            if config.DO_RANGE == False:
                i = config.SINGLE_FILE_NUM

            # Reset seeds for identical model behavior each iteration.
            random.seed(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)
            tf.random.set_seed(config.RANDOM_SEED)

            values_file = sorted(columns_dir.glob("*.txt"))[i - 1]
            x, _, batch = load_data(seq_file, values_file, max_seqs)
            y_raw = load_all_columns(columns_dir, values_file, config.MAX_SEQS)
        
            # Pass the variables to pca workflow.
            pca_values(
                x, y_raw, batch, 
                pca_components=config.PCA_COMPONENTS, 
                show_bounds=config.SHOW_BOUNDS, 
                std_multiplier=config.STD_MULTIPLIER, 
                frac=config.FRAC, 
                output_file=output_file, 
                pct_file=pct_file, 
                mode=config.OUTLIER_MODE, 
                do_pca=config.DO_PCA
            )

            if config.DO_RANGE == False:
                break
    
    # Take the average of an ensemble of models on one column.
    elif config.DO_ENSEMBLE:
        print("Ensemble mode enabled.")

        # Select the values file.
        for i in range(
            config.RANGE_START_FILE_NUM, config.RANGE_END_FILE_NUM + 1
        ):
            if config.DO_RANGE == False:
                i = config.SINGLE_FILE_NUM
            
            # Reset seeds for identical model behavior each iteration.
            random.seed(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)
            tf.random.set_seed(config.RANDOM_SEED)

            values_file = sorted(columns_dir.glob("*.txt"))[i - 1]
            x, y, batch = load_data(seq_file, values_file, max_seqs)
            col_name = values_file.name[:-4]

            # Pass the variables to ensemble workflow.
            ensemble(
                x, y, batch, 
                train_percentage=config.TRAIN_PERCENTAGE, 
                data_splits=config.DATA_SPLITS, 
                col_name=col_name, 
                show_bounds=config.SHOW_BOUNDS, 
                std_multiplier=config.STD_MULTIPLIER, 
                frac=config.FRAC, 
                output_file=output_file, 
                pct_file=pct_file, 
                mode=config.OUTLIER_MODE
            )

            if config.DO_RANGE == False:
                break

    # Train on part of one column and test on part of another.
    elif config.DO_DOUBLE_COLUMNS:
        print("Double columns mode enabled.")

        # Reset seeds before loading training data.
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

        # Load training data from DOUBLE_TRAIN_FILE.
        double_train_file = sorted(
            columns_dir.glob("*.txt")
        )[config.DOUBLE_TRAIN_FILE - 1]
        x, y_train, batch = load_data(seq_file, double_train_file, max_seqs)

        # Select the values file for testing.
        for i in range(
            config.RANGE_START_FILE_NUM, config.RANGE_END_FILE_NUM + 1
        ):
            if config.DO_RANGE == False:
                i = config.SINGLE_FILE_NUM

            # Reset seeds before loading testing data.
            random.seed(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)
            tf.random.set_seed(config.RANDOM_SEED)

            values_file2 = sorted(columns_dir.glob("*.txt"))[i - 1]
            _, y_test, batch_test = load_data(seq_file, values_file2, max_seqs)

            # Pass the variables to double_columns workflow.
            double_columns(
                x, y_train, y_test, batch, batch_test,
                train_file=config.DOUBLE_TRAIN_FILE, 
                test_file=i, 
                train_pctg=config.TRAIN_PERCENTAGE, 
                show_bounds=config.SHOW_BOUNDS, 
                std_multiplier=config.STD_MULTIPLIER, 
                frac=config.FRAC, 
                output_file=output_file, 
                pct_file=pct_file, 
                mode=config.OUTLIER_MODE
            )

            if config.DO_RANGE == False:
                break

    else:
        print("Error: No training workflow selected!")
        return False
    
    # After a workflow finishes completely, the program returns here.
    print("Training completed!")
    return True