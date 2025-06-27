import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.src.layers import LSTM
from scipy.stats import gaussian_kde, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

# Make sure any randomization is repeatable.
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


"""Note-taking section:

Divide into 10. Take 5/10, ten times, predict on 100
BIC! / AIC to measure overfitting
Try to visualize the overfit curve before BIC

Take two columns, split both in half. 1A, 1B, 2A, 2B.
Train on 1A, test on 2B. Repeat for all four diagonal directions.
Maybe try a mix of ensemble and single-model.
"""


# This one-hot encodes any sequence and returns it.
def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'T': [0, 0, 1, 0],
               'G': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([mapping[nuc] for nuc in sequence], dtype=np.float32)


# This one-hot-decodes any sequence back into letters and returns it.
def one_hot_decode(encoded_sequence):
    reverse_mapping = {(1, 0, 0, 0): 'A', (0, 1, 0, 0): 'C', (0, 0, 1, 0): 'T',
                       (0, 0, 0, 1): 'G', (0, 0, 0, 0): 'N'}
    decoded_sequence = ''.join(reverse_mapping.get(tuple(vec), 'N') 
                               for vec in encoded_sequence)
    return decoded_sequence


# This gets the relevant data and balances the ratio of values.
def load_data(fasta_file, values_file, max_seqs):
    sequences = []
    
    # Read sequences from FASTA file and join each to one line
    with open(fasta_file, 'r') as f:
        lines = f.read().splitlines()
        # FASTA has 1 useless header line followed by 7 sequence lines
        for i in range(0, len(lines), 8):
            full_sequence = ''.join(lines[i+1:i+8])
            sequences.append(full_sequence)

    # Read numerical values from a column
    values = np.loadtxt(values_file, dtype=np.float32)
    assert len(sequences) == len(values), "Mismatch between seq and value len."

    # Shuffle pairs and truncate to max_seqs
    all_pairs = list(zip(sequences, values))
    np.random.shuffle(all_pairs)
    final_pairs = all_pairs[:max_seqs]
    print(f"Unbalanced. Using {len(final_pairs)} total sequences.")

    # One-hot encode the sequences to finalize the pairs
    final_sequences = [one_hot_encode(seq) for (seq, val) in final_pairs]
    final_values = [val for (seq, val) in final_pairs]

    return np.array(final_sequences), np.array(final_values, dtype=np.float32)


# This loads every value file. Can use all of them or just one.
def load_all_column_targets(columns_dir, reference_file, max_seqs=None):    
    # Load the reference values to get the shuffle order
    reference_values = np.loadtxt(reference_file, dtype=np.float32)
    
    # Load all column values
    all_values = []
    for file in sorted(columns_dir.glob("*.txt")):
        values = np.loadtxt(file, dtype=np.float32)
        all_values.append(values)
    
    y_all = np.stack(all_values, axis=1)  # Shape: (n_samples, n_cols)
    
    # Shuffle pairs and truncate to max_seqs
    all_indices = np.arange(len(reference_values))
    np.random.shuffle(all_indices)
    final_indices = all_indices[:max_seqs]
    
    return y_all[final_indices]


# This constructs the layers of the model and returns it.
def build_model(input_shape, output_dim=1):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),

        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Bidirectional(LSTM(64, return_sequences=False)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(output_dim, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# This ensures the input numbers for ensemble() are valid.
def is_compatible(train_pct, data_splits):
    # The train_pct must be a multiple of (100 / data_splits) or bin_pct
    bin_pct = 100 / data_splits
    
    # Check for any conditions that would make the numbers invalid
    if train_pct % bin_pct != 0:
        print(f"train_pct ({train_pct}) must be a multiple of {bin_pct:.1f}.")
        return False
    elif train_pct <= 0 or train_pct > 100:
        print(f"train_pct ({train_pct}) must be between 1 and 100.")
        return False
    elif data_splits <= 0:
        print(f"data_splits ({data_splits}) must be positive.")
        return False
    
    print(f"{train_pct}% training is compatible with {data_splits} splits.")
    return True


# This makes a heatmap scatterplot with the data.
def plot_graph(y_true, y_pred, smse, col_name, file_num, show_bounds=True, 
               std_multiplier=2, frac=0.3, use_pca=False):
    plt.figure(figsize=(10, 6))

    # Calculate Pearson correlation for the graph title
    corr, _ = pearsonr(y_true, y_pred)

    xy = np.vstack([y_true, y_pred])
    density = gaussian_kde(xy)(xy)
    plt.scatter(y_true, y_pred, c=density, cmap="RdYlGn_r", edgecolors="none", 
                alpha=0.6, s=4)
    plt.colorbar(label="Density")

    # Shows the LOESS curve and boundary lines on the graph
    if show_bounds:
        # Calculate and plot the LOESS curve
        loess_result = lowess(y_pred, y_true, frac=frac)
        x_loess, y_loess = loess_result[:, 0], loess_result[:, 1]

        plt.plot(x_loess, y_loess, color='black', linewidth=1, label='LOESS')

        # Interpolate to get predicted y on the LOESS line at x points
        loess_interp = np.interp(y_true, x_loess, y_loess)
        residuals = y_pred - loess_interp
        std_dev = np.std(residuals)

        # Calculate and plot the bounds 
        upper = y_loess + std_multiplier * std_dev
        lower = y_loess - std_multiplier * std_dev

        plt.plot(x_loess, upper, color='gray', linewidth=0.5, linestyle='--', 
                 label=f'+{std_multiplier}σ')
        plt.plot(x_loess, lower, color='gray', linewidth=0.5, linestyle='--', 
                 label=f'-{std_multiplier}σ')

    # Labels
    xlabel = 'True Values (PCA)' if use_pca else 'True Values'
    ylabel = 'Predicted Values (PCA)' if use_pca else 'Predicted Values'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{col_name} (r={corr:.3f}) (SMSE: {smse:.4f})', fontsize=11)
    plt.legend()

    # Save the figure to Downloads folder
    save_dir = Path.home() / 'Downloads'
    save_name = f"{col_name}.png"
    save_path = save_dir / save_name
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.close()


# This outputs the outliers to a new file.
def simple_log(outliers, output_file):
    # Save outputs to 'outliers' folder
    outliers_dir = Path.cwd() / 'outliers'
    if not outliers_dir.exists():
        print(f"'outliers' directory not found at {outliers_dir}")
        return
    
    output_file_path = outliers_dir / output_file
    with open(output_file_path, 'w') as f:
        for sequence, category in outliers:
            f.write(f"{sequence}\t{category}\n")


# This updates an existing file with the new outlier information.
def complex_log(outliers, output_file, column_name):
    outlier_dict = {}

    # Load existing data if it exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    sequence = parts[0]
                    count = int(parts[1])
                    col_and_category = parts[2:]
                    outlier_dict[sequence] = [count, col_and_category]

    # Update the outlier_dict
    for sequence, category in outliers:
        if sequence not in outlier_dict:
            outlier_dict[sequence] = [1, [f"{column_name}:{category}"]]
        else:
            outlier_dict[sequence][0] += 1
            outlier_dict[sequence][1].append(f"{column_name}:{category}")

    # Write updated data
    with open(output_file, 'w') as f:
        for sequence, (count, col_and_category) in outlier_dict.items():
            f.write(f"{sequence}\t{count}\t" + 
            "\t".join(col_and_category) + "\n")


# This detects and logs outliers based on deviation from LOESS.
def detect_outliers(y_true, y_pred, sequences, output_file, pct_file, 
                    col_name, std_multiplier, frac, mode, use_pca):
    if mode == 'off':
        print("Outlier detection skipped (mode='off').")
        return
    
    if use_pca:
        mode = 'simple'
        print("PCA mode enabled - using simple outlier detection.")
    
    # Fit LOESS curve    
    loess_sorted = lowess(y_pred, y_true, frac=frac)
    x_loess, y_loess = loess_sorted[:, 0], loess_sorted[:, 1]
    fitted = np.interp(y_true, x_loess, y_loess)
    residuals = y_pred - fitted

    # Calculate standard deviation of residuals
    std_dev = np.std(residuals)
    upper_bound = fitted + std_multiplier * std_dev
    lower_bound = fitted - std_multiplier * std_dev

    # Identify the outliers
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

    # Print percentage stats
    total_count = len(y_true)
    outlier_count = len(outliers)
    percent = (outlier_count / total_count) * 100
    print(f"Outliers detected: {outlier_count}/{total_count} ({percent:.2f}%)")

    # Write percentage to a outlier_percents.txt file
    with open(pct_file, 'a') as f:
        f.write(f"{col_name} Outliers detected: {outlier_count}/{total_count} "
                f"({percent:.2f}%)\n")

    # Select the method of logging
    if mode == 'simple' or use_pca:
        simple_log(outliers, f'{col_name}_outliers.txt')
    elif mode == 'complex':
        complex_log(outliers, output_file, col_name)
    elif mode == 'both':
        simple_log(outliers, f'{col_name}_outliers.txt')
        complex_log(outliers, output_file, col_name)
    else:
        raise ValueError("Mode must be 'simple', 'complex', or 'both'.")
    
    print("Outliers logged.")


# This applies PCA to values and predicts values.
def pca_values(x, y_raw, file_num, pca_components, show_bounds, std_multiplier, frac, use_pca, complex_output_file, pct_file, mode):
    print(f"Loaded {x.shape[0]} sequences and {y_raw.shape[0]} value rows")
    
    # Scale PCA components for multi-output learning
    scaler_pca = StandardScaler()
    y_raw_scaled = scaler_pca.fit_transform(y_raw)

    # Use entire dataset to create PCA components
    pca = PCA(n_components=pca_components)
    # Scale before fit transform
    y_all_scaled = pca.fit_transform(y_raw)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    print("After scaling - all components should have std ~1.0")
    print(f"Scaled range: {np.min(y_raw_scaled):.3f} to {np.max(y_raw_scaled):.3f}")

    # Save each PCA component to 'components' folder
    components_dir = Path.cwd() / 'components'
    for i in range(pca_components):
        comp_file = components_dir / f'component {i}.txt'
        np.savetxt(comp_file, y_all_scaled[:, i], fmt='%.6f')
    print(f"Saved {pca_components} PCA component files to {components_dir}")
    
    # Loop over each component, train and evaluate a single-output model
    for i in range(pca_components):
        print(f"\nTraining for PCA Component {i}")
        y_component = y_all_scaled[:, i]
        x_train, x_test, y_train, y_test = train_test_split(x, y_component, test_size=0.2, random_state=42)

        # Build model
        model = build_model((x.shape[1], x.shape[2]), 1)  # (500, 4)

        # Train the model
        model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=1)
        predictions = model.predict(x_test).flatten()

        # Debug: Check prediction variability
        print(f"Prediction std: {np.std(predictions):.4f}")
        print(f"y_test std: {np.std(y_test):.4f}")

        # Transform predictions back to original scale (original units)
        #predictions = scaler_pca.inverse_transform(predictions)
        #y_test = scaler_pca.inverse_transform(y_test)

        # Calculate SMSE
        smse = mean_squared_error(y_test, predictions, squared=False)
        print(f"Component {i} SMSE: {smse}")

        # Create graph and outlier files
        col_name = f"Val-PCA-Component-{i+1}"
        plot_graph(y_test, predictions, smse, col_name, file_num, show_bounds, std_multiplier, frac, use_pca)

        # Create separate outlier file for each component pair
        detect_outliers(y_test, predictions, x_test, complex_output_file, pct_file, col_name, std_multiplier, frac, mode, use_pca=True)


# This ensemble approach averages predictions from multiple models.
def ensemble(x, y, col_name, train_percentage, data_splits, show_bounds, std_multiplier, frac, complex_output_file, pct_file, mode, file_num):
    # Train on overlapping continuous segments, test on full dataset

    # Compatibility check
    if not is_compatible(train_percentage, data_splits):
        print(f"Incompatible parameters: train_percentage={train_percentage}, data_splits={data_splits}")
        return
    
    # Save outputs to 'ensemble' folder
    ensemble_dir = Path.cwd() / 'ensemble'
    if not ensemble_dir.exists():
        print(f"'outliers' directory not found at {ensemble_dir}")
        return
    
    # Split the data into bins to iterate through
    total_samples = len(x)
    bin_size = total_samples // data_splits
    bins_to_use = int(data_splits * train_percentage / 100)
    n_models = data_splits

    print(f"Total samples: {total_samples:,}")
    print(f"Bin size: {bin_size:,}")
    print(f"Bins to use per model: {bins_to_use}")
    print(f"Number of models: {n_models}")

    # Calculate training range percentages for each model
    train_ranges = []
    step_size = 100 / data_splits  # Step size as percentage

    for model_idx in range(n_models):
        start_pct = (model_idx * step_size) % 100
        end_pct = (start_pct + train_percentage)
        start_idx = int(total_samples * start_pct / 100)
        
        # Doesn't wrap around to the start
        if end_pct <= 100:
            end_idx = int(total_samples * end_pct / 100)
            train_indices = list(range(start_idx, end_idx))
            range_desc = f"{start_pct:.1f}-{end_pct:.1f}"
    
        # Wraps around to the start
        else:
            end_pct_wrapped = end_pct - 100
            end_idx = int(total_samples * end_pct_wrapped / 100)
            train_indices = (list(range(start_idx, total_samples)) + 
                             list(range(0, end_idx)))
            range_desc = f"{start_pct:.1f}-100, 0-{end_pct_wrapped:.1f}"
        
        train_ranges.append(train_indices)

        # Print readable range description
        print(f"Model {model_idx + 1}: {range_desc}% "
              f"({len(train_indices):,} samples)")
        
    # Train models and collect predictions on full dataset
    all_predictions = []
    for model_idx, train_indices in enumerate(train_ranges):
        print(f"Training Model {model_idx + 1}/{n_models} ...")
        
        x_train = x[train_indices]
        y_train = y[train_indices]
        
        model = build_model((500, 4), 1)
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
        
        full_predictions = model.predict(x, verbose=0).flatten()
        all_predictions.append(full_predictions)

    print(f"Averaging predictions from {n_models} models...")
    
    # Convert to array of shape (n_models, n_samples)
    predictions_array = np.array(all_predictions)
    averaged_predictions = np.mean(predictions_array, axis=0)

    # Calculate metrics
    final_corr = np.corrcoef(y, averaged_predictions)[0, 1]
    final_smse = mean_squared_error(y, averaged_predictions, squared=False)
    
    print(f"Final correlation: {final_corr:.4f}")
    print(f"Final SMSE: {final_smse:.4f}")
    print(f"Prediction std: {np.std(averaged_predictions):.4f}")
    
    # Save results
    method_name = f"ensemble{train_percentage}p{data_splits}s"
    pred_file = ensemble_dir / f'{method_name}_{col_name}.txt'
    np.savetxt(pred_file, averaged_predictions, fmt='%.6f')
    print(f"Saved predictions: {pred_file}")
    
    # Create graph and outlier files
    print(f"Creating graph and outlier files...")
    final_col_name = f"ensemble{train_percentage}p{data_splits}s-{col_name}"
    
    plot_graph(y, averaged_predictions, final_smse, final_col_name, file_num,
               show_bounds, std_multiplier, frac, use_pca=False)
    detect_outliers(y, averaged_predictions, x, complex_output_file, pct_file,
                    final_col_name, std_multiplier, frac, mode, use_pca=False)


# This trains on and predicts values for one specific column, no PCA.
def no_pca(do_kfold, x, y, file_num, col_name, show_bounds, std_multiplier, frac, complex_output_file, pct_file, mode):
    if do_kfold:
        print(f"PCA disabled - using K-fold cross-validation on one file.")
    else:
        print(f"PCA disabled - single run on column {col_name}.")

    # Save all values for final combined graph
    all_y_true = []
    all_y_pred = []
    all_sequences = []

    # Begin fold loop
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_idx, test_idx in kf.split(x):
        if do_kfold:
            print(f"Fold {fold} begin: ")

        # Prepare data for the model
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model((500, 4))

        # Train the model
        model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
        predictions = model.predict(x_test).flatten()

        # Calculate and print SMSE
        smse = mean_squared_error(y_test, predictions, squared=False)
        if do_kfold:
            print(f"Fold {fold} SMSE: {smse}")
        else:
            print(f"Single run SMSE: {smse}")

        if do_kfold:
            # Plot graph for individual fold
            plot_graph(y_test, predictions, smse, f"{col_name} - Fold {fold}", file_num, show_bounds, std_multiplier, frac)

            # Detect and log outliers
            detect_outliers(y_test, predictions, x_test, complex_output_file, pct_file, f"{col_name} - Fold {fold}", std_multiplier, frac, mode)

            # Store for the combined graph
            all_y_true.extend(y_test)
            all_y_pred.extend(predictions)
            all_sequences.extend(x_test)

            fold += 1
        
        else:
            # Graph and outliers for single run and end loop
            plot_graph(y_test, predictions, smse, f"{col_name}", file_num, show_bounds, std_multiplier, frac)
            detect_outliers(y_test, predictions, x_test, complex_output_file, pct_file, f"{col_name}", std_multiplier, frac, mode)

            return

    # Final combined graph after all folds
    total_smse = mean_squared_error(all_y_true, all_y_pred, squared=False)
    plot_graph(np.array(all_y_true), np.array(all_y_pred), total_smse, f"{col_name} - Combined 5-Fold", file_num, show_bounds, std_multiplier, frac)

    # Final outlier after all folds
    detect_outliers(all_y_true, all_y_pred, all_sequences, complex_output_file, pct_file, f"{col_name} - Combined", std_multiplier, frac, mode='both')


# This is the program's main function.
def main():
    # Configure variables per session
    file_num = 0  # 0-49
    use_pca = False
    transpose = False
    do_ensemble = True
    do_kfold = False

    # Not often changed, but here for convenience
    max_seqs = 157455  # Total seqs: 157455
    pca_components = 4
    train_percentage = 80
    data_splits = 5
    show_bounds = True
    std_multiplier = 2
    frac = 0.3  # 0-1. Affects smoothness and sensitivity of curve
    mode = 'both'  # Outlier logging. 'simple', 'complex', 'both', 'off'

    # Access 'columns' folder which holds the value files
    cwd = Path.cwd()
    columns_dir = cwd / 'columns'
    if not columns_dir.exists():
        print(f"'columns' directory not found at {columns_dir}")
        return
    
    # Get and load seq file and value files
    seq_file = cwd / 'seqs.fa'
    values_file = sorted(columns_dir.glob("*.txt"))[file_num]
    x, y = load_data(seq_file, values_file, max_seqs=max_seqs)
        
    # Prepare files for detect_outliers() function
    col_name = values_file.name[:-4]
    complex_output_file = cwd / 'outliers.txt'
    pct_file = cwd / 'outlier_percents.txt'

    if use_pca:
        print("PCA enabled - loading full dataset.")
        y_raw = load_all_column_targets(columns_dir, values_file, max_seqs=max_seqs)  # ALL columns

        if transpose:
            print("Transposed mode not yet integrated.")
            return
        else:
            print("Applying PCA to values (original mode).")
            pca_values(x, y_raw, file_num, pca_components, show_bounds, std_multiplier, frac, use_pca, complex_output_file, pct_file, mode)
    
    elif do_ensemble:
        print("Ensemble mode enabled.")
        ensemble(x, y, col_name, train_percentage, data_splits, show_bounds, std_multiplier, frac, complex_output_file, pct_file, mode, file_num)

    else:
        # No PCA or ensemble use, train on one column's values only
        no_pca(do_kfold, x, y, file_num, col_name, show_bounds, std_multiplier, frac, complex_output_file, pct_file, mode)

    print("Program is finished.")


# Start the program.
if __name__ == '__main__':
    main()
