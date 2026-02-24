import os
from pathlib import Path

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from src import config
from src.data_loading import one_hot_decode


# This outputs the outliers to a new file.
def _simple_log(outliers, output_file):
    outliers_dir = Path.cwd() / 'outputs' / 'simple_outliers'
    output_file_path = outliers_dir / output_file
    
    try:
        with open(output_file_path, 'w') as f:
            for sequence, category in outliers:
                f.write(f"{sequence}\t{category}\n")
    except Exception as e:
        print(f"Error saving simple outliers: {e}")


# This updates an existing file with the new outlier information.
def _complex_log(outliers, output_file, column_name):
    outlier_dict = {}

    # Load existing data if it exists.
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    sequence = parts[0]
                    count = int(parts[1])
                    col_and_category = parts[2:]
                    outlier_dict[sequence] = [count, col_and_category]

    # Update the outlier_dict.
    for sequence, category in outliers:
        if sequence not in outlier_dict:
            outlier_dict[sequence] = [1, [f"{column_name}:{category}"]]
        else:
            outlier_dict[sequence][0] += 1
            outlier_dict[sequence][1].append(f"{column_name}:{category}")

    # Write updated data.
    with open(output_file, 'w') as f:
        for sequence, (count, col_and_category) in outlier_dict.items():
            f.write(
                f"{sequence}\t{count}\t" + "\t".join(col_and_category) + "\n"
            )


# This detects and logs outliers based on deviation from LOESS.
def detect_outliers(
        sequences, y_true, y_pred, output_file, pct_file, col_name, mode
):
    if mode.lower() == 'off':
        print("Outlier detection skipped (mode='off').")
        return
    
    if config.DO_PCA == True:
        mode = 'simple'
        print("PCA mode enabled - using simple outlier detection.")
    
    # Fit LOESS curve.  
    loess_sorted = lowess(y_pred, y_true, frac=config.FRAC)
    x_loess, y_loess = loess_sorted[:, 0], loess_sorted[:, 1]
    fitted = np.interp(y_true, x_loess, y_loess)
    residuals = y_pred - fitted

    # Calculate standard deviation of residuals.
    std_dev = np.std(residuals)
    std_multiplier = config.STD_MULTIPLIER
    upper_bound = fitted + std_multiplier * std_dev
    lower_bound = fitted - std_multiplier * std_dev

    # Identify the outliers.
    outliers = []
    for idx, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
        if pred_val > upper_bound[idx]:
            category = 'overfitted'
        elif pred_val < lower_bound[idx]:
            category = 'underfitted'
        else:
            continue
        sequence = one_hot_decode(sequences[idx])
        outliers.append((sequence, category))

    # Print percentage stats.
    total_count = len(y_true)
    outlier_count = len(outliers)
    percent = (outlier_count / total_count) * 100
    print(f"Outliers detected: {outlier_count}/{total_count} ({percent:.2f}%)")

    # Write percentage to outlier_percents.txt file.
    with open(pct_file, 'a') as f:
        f.write(f"{col_name} Outliers detected: {outlier_count}/{total_count} "
                f"({percent:.2f}%)\n")

    # Select the method of logging.
    if mode.lower() == 'simple' or mode.lower() == 'both':
        _simple_log(outliers, f'{col_name}_outliers.txt')
    elif mode.lower() == 'complex' or mode.lower() == 'both':
        _complex_log(outliers, output_file, col_name)
    else:
        raise ValueError("Mode must be 'simple', 'complex', 'both', or 'off'.")
    
    print("Outliers logged.")