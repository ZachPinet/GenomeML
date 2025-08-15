import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from src import config
from src.data_loading import load_data, load_all_columns
from src.workflows.single_columns import single_column
from src.workflows.pca import pca_values
from src.workflows.ensemble import ensemble
from src.workflows.double_columns import double_columns

# Run the appropriate training workflow based on current config values
def run_training():
    # Make sure any randomization is repeatable.
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    # Access 'columns' folder which holds the value files
    cwd = Path.cwd()
    columns_dir = cwd / 'inputs' / 'columns'
    seq_file = cwd / 'inputs' / 'seqs.fa'
    
    # Get and load seq file and value files
    values_file = sorted(columns_dir.glob("*.txt"))[config.FILE_NUM - 1]
    x, y = load_data(seq_file, values_file, max_seqs=config.MAX_SEQS)
        
    # Prepare files for detect_outliers() function
    col_name = values_file.name[:-4]
    output_file = cwd / 'outputs' / 'outliers.txt'
    pct_file = cwd / 'outputs' / 'outlier_percents.txt'

    print(config.DOUBLE_TRAIN_FILE)
    print(config.FILE_NUM)
    print(config.FILE_NUM2)

    # Test and train on one column's values only
    if config.DO_SINGLE_COLUMN:
        print("Single column mode enabled.")

        for i in range(config.FILE_NUM, config.FILE_NUM2 + 1):
            # Reset seeds for identical model behavior each iteration
            random.seed(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)
            tf.random.set_seed(config.RANDOM_SEED)

            curr_file = i
            values_file = sorted(columns_dir.glob("*.txt"))[curr_file]
            x, y = load_data(seq_file, values_file, max_seqs=config.MAX_SEQS)

            single_column(
                x, y, 
                kfold=config.KFOLD, 
                col_name=col_name + 1, 
                show_bounds=config.SHOW_BOUNDS, 
                std_multiplier=config.STD_MULTIPLIER, 
                frac=config.FRAC, 
                output_file="outliers.txt", 
                pct_file="outliers_pct.txt", 
                mode=config.MODE
            )

            if config.DO_RANGE == False:
                break

    # Use PCA on all columns
    elif config.DO_PCA:
        print("PCA mode enabled - loading full dataset.")
        values_file = sorted(columns_dir.glob("*.txt"))[config.FILE_NUM]
        x, _ = load_data(seq_file, values_file, max_seqs=config.MAX_SEQS)
        y_raw = load_all_columns(columns_dir, values_file, config.MAX_SEQS)

        if config.TRANSPOSE:
            print("Transposed mode not integrated. Running normal PCA")
        
        pca_values(
            x, y_raw, 
            pca_components=config.PCA_COMPONENTS, 
            show_bounds=config.SHOW_BOUNDS, 
            std_multiplier=config.STD_MULTIPLIER, 
            frac=config.FRAC, 
            output_file="outliers.txt", 
            pct_file="outliers_pct.txt", 
            mode=config.MODE, 
            do_pca=config.DO_PCA
        )
    
    # Do an ensemble of models on one column
    elif config.DO_ENSEMBLE:
        print("Ensemble mode enabled.")

        for i in range(config.FILE_NUM, config.FILE_NUM2 + 1):
            # Reset seeds for identical model behavior each iteration
            random.seed(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)
            tf.random.set_seed(config.RANDOM_SEED)

            curr_file = i
            values_file = sorted(columns_dir.glob("*.txt"))[curr_file]
            x, y = load_data(seq_file, values_file, max_seqs=config.MAX_SEQS)

            ensemble(
                x, y, 
                train_percentage=config.TRAIN_PERCENTAGE, 
                data_splits=config.DATA_SPLITS, 
                col_name=col_name + 1, 
                show_bounds=config.SHOW_BOUNDS, 
                std_multiplier=config.STD_MULTIPLIER, 
                frac=config.FRAC, 
                output_file="outliers.txt", 
                pct_file="outliers_pct.txt", 
                mode=config.MODE
            )

            if config.DO_RANGE == False:
                break

    # Train on part of one column and test on part of another
    elif config.DO_DOUBLE_COLUMNS:
        print("Double columns mode enabled.")

        # Reset seeds before loading training data
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

        # Load training data from DOUBLE_TRAIN_FILE
        double_train_file = sorted(
            columns_dir.glob("*.txt")
        )[config.DOUBLE_TRAIN_FILE - 1]
        x, y_train = load_data(
            seq_file, double_train_file, max_seqs=config.MAX_SEQS
        )

        for i in range(config.FILE_NUM, config.FILE_NUM2 + 1):
            # Reset seeds before loading testing data
            random.seed(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)
            tf.random.set_seed(config.RANDOM_SEED)

            # Use the same shuffling as training data
            values_file2 = sorted(columns_dir.glob("*.txt"))[i - 1]
            _, y_test = load_data(
                seq_file, values_file2, max_seqs=config.MAX_SEQS
            )

            double_columns(
                x, y_train, y_test, 
                train_file=config.DOUBLE_TRAIN_FILE, 
                test_file=i, 
                train_pctg=config.TRAIN_PERCENTAGE, 
                show_bounds=config.SHOW_BOUNDS, 
                std_multiplier=config.STD_MULTIPLIER, 
                frac=config.FRAC, 
                output_file="outliers.txt", 
                pct_file="outliers_pct.txt", 
                mode=config.MODE
            )

            if config.DO_RANGE == False:
                break

    else:
        print("Error: No training workflow selected!")
        return False
    
    print("Training completed!")
    return True