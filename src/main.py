import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config
from .data_loading import load_data, load_all_columns
from .workflows.double_columns import double_columns
from .workflows.ensemble import ensemble
from .workflows.pca import pca_values
from .workflows.single_columns import single_column


# This is the program's main function.
def main():
    # Make sure any randomization is repeatable.
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    # Access 'columns' folder which holds the value files
    cwd = Path.cwd()
    columns_dir = cwd / 'inputs' / 'columns'
    if not columns_dir.exists():
        print(f"'columns' directory not found at {columns_dir}")
        return
    
    # Get and load seq file and value files
    seq_file = cwd / 'inputs' / 'seqs.fa'
    values_file = sorted(columns_dir.glob("*.txt"))[config.FILE_NUM]
    x, y = load_data(seq_file, values_file, max_seqs=config.MAX_SEQS)
        
    # Prepare files for detect_outliers() function
    col_name = values_file.name[:-4]
    output_file = cwd / 'outputs' / 'outliers.txt'
    pct_file = cwd / 'outputs' / 'outlier_percents.txt'

    if config.DO_SINGLE_COLUMN:
        # No PCA or ensemble use, train on one column's values only
        single_column(
            x, y, config.DO_KFOLD, col_name, 
            config.SHOW_BOUNDS, config.STD_MULTIPLIER, config.FRAC, 
            output_file, pct_file, config.MODE
        )

    if config.USE_PCA:
        print("PCA enabled - loading full dataset.")
        y_raw = load_all_columns(columns_dir, values_file, config.MAX_SEQS)

        if config.TRANSPOSE:
            print("Transposed mode not integrated. Running normal PCA")
        
        print("Applying PCA to values (original mode).")
        pca_values(
            x, y_raw, config.PCA_COMPONENTS, 
            config.SHOW_BOUNDS, config.STD_MULTIPLIER, config.FRAC, 
            output_file, pct_file, config.MODE, config.USE_PCA
        )
    
    if config.DO_ENSEMBLE:
        print("Ensemble mode enabled.")
        ensemble(
            x, y, config.TRAIN_PERCENTAGE, config.DATA_SPLITS, col_name, 
            config.SHOW_BOUNDS, config.STD_MULTIPLIER, config.FRAC, 
            output_file, pct_file, config.MODE
        )

    if config.DO_DOUBLE_COLUMNS:
        print("Double columns mode enabled.")

        for i in range(30, 50):
            file_num2 = i

            # Load second column's values using the same sequences
            np.random.seed(42)  # Reset seed for same shuffling as y
            values_file2 = sorted(columns_dir.glob("*.txt"))[file_num2]
            _, y2 = load_data(seq_file, values_file2, max_seqs=config.MAX_SEQS)

            double_columns(
                x, y, y2, config.FILE_NUM, file_num2, config.TRAIN_PERCENTAGE, 
                config.SHOW_BOUNDS, config.STD_MULTIPLIER, config.FRAC, 
                output_file, pct_file, config.MODE
            )

    print("Program is finished.")