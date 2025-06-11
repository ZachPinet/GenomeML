import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess
import tensorflow as tf
from keras.src.layers import LSTM
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make sure any randomization is repeatable.
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# This one-hot encodes any sequence and returns it.
def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'T': [0, 0, 1, 0], 'G': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([mapping[nuc] for nuc in sequence], dtype=np.float32)


# This one-hot-decodes any sequence back into letters and returns it.
def one_hot_decode(encoded_sequence):
    reverse_mapping = {(1, 0, 0, 0): 'A', (0, 1, 0, 0): 'C', (0, 0, 1, 0): 'T', (0, 0, 0, 1): 'G', (0, 0, 0, 0): 'N'}
    
    decoded_sequence = ''.join(
        reverse_mapping.get(tuple(vec), 'N') for vec in encoded_sequence
    )
    return decoded_sequence


# This gets the relevant data and balances the ratio of values.
def load_data(fasta_file, values_file, max_seqs, balance=False):
    sequences = []
    
    # Read and reconstruct sequences from FASTA file
    with open(fasta_file, 'r') as f:
        lines = f.read().splitlines()
        # FASTA has 1 useless header line followed by 7 sequence lines
        for i in range(0, len(lines), 8):
            full_sequence = ''.join(lines[i+1:i+8])  # Join to 1 line
            sequences.append(full_sequence)

    # Read numerical values from a column
    values = np.loadtxt(values_file, dtype=np.float32)
    assert len(sequences) == len(values), "Mismatch between seq and value len."
    
    # Balance the data if balancing is enabled
    if balance:
        """
        Balancing is currently not optimal. Bins ensure equal counts, but not value spread, so there are still disproportionate amounts of similar values.
        """
        # Compute quantile boundaries
        q1, q2, q3 = np.percentile(values, [25, 50, 75])

        # Bin into 4 equal-frequency groups
        bins = {0: [], 1: [], 2: [], 3: []}
        for seq, val in zip(sequences, values):
            if val <= q1:
                bins[0].append((seq, val))
            elif val <= q2:
                bins[1].append((seq, val))
            elif val <= q3:
                bins[2].append((seq, val))
            else:
                bins[3].append((seq, val))

        # Determine the size of each bin
        min_count = min(len(b) for b in bins.values())
        min_count = min(min_count, max_seqs // 4)
        print(f"Seqs balanced. {min_count} per bin (Total: {min_count * 4}).")

        # Sample and combine from each bin
        final_pairs = []
        for i in range(4):
            selected = np.random.choice(len(bins[i]), size=min_count, replace=False)
            final_pairs.extend([bins[i][j] for j in selected])
        np.random.shuffle(final_pairs)

    # Don't balance, just shuffle and truncate to max_seqs
    else:
        all_pairs = list(zip(sequences, values))
        np.random.shuffle(all_pairs)
        final_pairs = all_pairs[:max_seqs]
        print(f"Unbalanced. Using {len(final_pairs)} total sequences.")

    # One-hot encode the sequences to finalize the pairs
    final_sequences = [one_hot_encode(seq) for (seq, val) in final_pairs]
    final_values = [val for (seq, val) in final_pairs]

    return np.array(final_sequences), np.array(final_values, dtype=np.float32)


# This loads every value file. Can use all of them or just one.
def load_all_column_targets(columns_dir, reference_file, max_seqs=None, balance=False):
    """Load all column targets with the same shuffling as load_data()"""
    
    # Load the reference values to get the shuffle order
    reference_values = np.loadtxt(reference_file, dtype=np.float32)
    
    # Load all column values
    all_values = []
    for file in sorted(columns_dir.glob("*.txt")):
        values = np.loadtxt(file, dtype=np.float32)
        all_values.append(values)
    
    y_all = np.stack(all_values, axis=1)  # Shape: (n_samples, n_columns)
    
    # Apply the SAME balancing/shuffling logic as load_data()
    if balance:
        # Same balancing logic as in load_data()
        q1, q2, q3 = np.percentile(reference_values, [25, 50, 75])
        
        bins = {0: [], 1: [], 2: [], 3: []}
        for idx, val in enumerate(reference_values):
            if val <= q1:
                bins[0].append(idx)
            elif val <= q2:
                bins[1].append(idx)
            elif val <= q3:
                bins[2].append(idx)
            else:
                bins[3].append(idx)
        
        min_count = min(len(b) for b in bins.values())
        min_count = min(min_count, max_seqs // 4)
        
        final_indices = []
        for i in range(4):
            selected = np.random.choice(bins[i], size=min_count, replace=False)
            final_indices.extend(selected)
        np.random.shuffle(final_indices)
        
    else:
        # Same shuffling logic as in load_data()
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


# This builds a model specifically designed for PCA sequence prediction.
def build_pca_seq_model(input_shape, output_dim=1):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),

        # More aggressive feature extraction for PCA components
        tf.keras.layers.Conv1D(128, kernel_size=7, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),

        # Larger dense layers for complex PCA relationships
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(output_dim, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
    

# This pairs the best correlations between true and predicted PCA.
def find_correlations(y_test, predictions, pca_components):
    print("Calculating component correlations...")
    correlations = np.zeros((pca_components, pca_components))
    
    for i in range(pca_components):  # True value components
        for j in range(pca_components):  # Prediction components
            corr, _ = pearsonr(y_test[:, i], predictions[:, j])
            correlations[i, j] = abs(corr)  # Absolute correlation
    
    print("Correlation matrix (rows=true, cols=prediction):")
    print(correlations)
    
    # Find the best matches for components based on correlations
    used_pred_components = set()
    component_pairs = []
    
    # Sort all correlations by strongest first
    correlation_list = []
    for i in range(pca_components):
        for j in range(pca_components):
            correlation_list.append((correlations[i, j], i, j))
    
    correlation_list.sort(reverse=True)
    
    # Pick next strongest correlation that doesn't conflict
    for corr_val, true_comp, pred_comp in correlation_list:
        if true_comp not in [pair[1] for pair in component_pairs] and pred_comp not in used_pred_components:
            component_pairs.append((corr_val, true_comp, pred_comp))
            used_pred_components.add(pred_comp)
            print(f"Matched True Component {true_comp} with Prediction Component {pred_comp} (r={corr_val:.4f})")
    
    return component_pairs


# This makes a heatmap scatterplot with the data.
def plot_graph(y_true, y_pred, smse, col_name, file_num, show_bounds=True, std_multiplier=2, frac=0.3, use_pca=False):    
    plt.figure(figsize=(10, 6))

    # Calculate Pearson correlation for the graph title
    corr, _ = pearsonr(y_true, y_pred)

    xy = np.vstack([y_true, y_pred])
    density = gaussian_kde(xy)(xy)

    plt.scatter(y_true, y_pred, c=density, cmap="RdYlGn_r", edgecolors="none", alpha=0.6, s=4)
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

        plt.plot(x_loess, upper, color='gray', linewidth=0.5, linestyle='--', label=f'+{std_multiplier}σ')
        plt.plot(x_loess, lower, color='gray', linewidth=0.5, linestyle='--', label=f'-{std_multiplier}σ')

    # Labels
    xlabel = 'True Values (PCA)' if use_pca else 'True Values'
    ylabel = 'Predicted Values (PCA)' if use_pca else 'Predicted Values'
    title = 'PCA' if use_pca else 'Actual'

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{col_name} - Pred vs {title} Values (r={corr:.3f}) (SMSE: {smse:.4f})', fontsize=11)
    plt.legend()

    # Save the figure to Downloads folder
    save_dir = Path.home() / 'Downloads'
    save_name = f"{col_name}.png" if use_pca else f"Test {file_num} - {col_name}.png"
    save_path = save_dir / save_name

    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# This outputs the outliers to a new file.
def simple_log(outliers, output_file):
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
            f.write(f"{sequence}\t{count}\t" + "\t".join(col_and_category) + "\n")


# This detects and logs outliers based on deviation from LOESS.
def detect_outliers(y_true, y_pred, sequences, output_file, percent_file, col_name, std_multiplier=2, frac=0.3, mode='simple', use_pca=False):
    if mode == 'off':
        print("Outlier detection skipped (mode='off').")
        return
    
    # deprecated Flatten input if it's multidimensional (for PCA predictions)
    if use_pca:
        #y_true = np.ravel(y_true)
        #y_pred = np.ravel(y_pred)
        mode = 'simple'  # Force simple mode in PCA
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
    with open(percent_file, 'a') as f:
        f.write(f"{col_name} Outliers detected: {outlier_count}/{total_count} ({percent:.2f}%)\n")

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


# This applies PCA to sequences and predicts single target values.
def pca_sequences(x, y_raw, file_num, pca_components, show_bounds, std_multiplier, frac, use_pca, complex_output_file, percent_file, mode):
    # Flatten sequences for PCA: (n_samples, 500*4) -> (n_samples, 2000)
    x_flattened = x.reshape(x.shape[0], -1)
            
    # Apply PCA to sequences
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_flattened)
            
    pca_x = PCA(n_components=pca_components)
    x_pca = pca_x.fit_transform(x_scaled)
            
    print(f"PCA explained variance ratio: {pca_x.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca_x.explained_variance_ratio_):.4f}")  # Ask about this
            
    # Use original target values (choose one column for now)
    y_target = y_raw[:, file_num]
            
    # Center and scale the target values
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1)).flatten()
            
    col_name = f"Seq-PCA-{pca_components}D"
    model = build_pca_seq_model((pca_components,), 1)
            
    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y_scaled, test_size=0.2, random_state=42)
    x_test_orig, _, _, _ = train_test_split(x, y_scaled, test_size=0.2, random_state=42)
            
    # Train the model
    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
    predictions = model.predict(x_test).flatten()
            
    # Transform predictions back to original scale
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_orig = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
    smse = mean_squared_error(y_test_orig, predictions_orig, squared=False)
    print(f"Sequence PCA Mode SMSE: {smse}")
            
    # Create single plot and outlier detection for sequence PCA
    plot_graph(y_test_orig, predictions_orig, smse, col_name, file_num, show_bounds, std_multiplier, frac, use_pca)
    detect_outliers(y_test_orig, predictions_orig, x_test_orig, complex_output_file, percent_file, col_name, std_multiplier, frac, mode, use_pca=True)


# This applies PCA to values and predicts values.
def pca_values(x, y_raw, file_num, pca_components, show_bounds, std_multiplier, frac, use_pca, complex_output_file, percent_file, mode):
    # Center and scale the target values before PCA
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_raw)
    
    # Use entire dataset for PCA
    pca = PCA(n_components=pca_components)
    y_all = pca.fit_transform(y_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    col_name = f"Val-PCA-{pca_components}D"
    model = build_model((500, 4), pca_components)

    # Prepare data for the model
    x_train, x_test, y_train, y_test = train_test_split(x, y_all, test_size=0.2, random_state=42)

    # Train the model
    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_test, y_test))
    predictions = model.predict(x_test)

    smse = mean_squared_error(y_test, predictions, squared=False)
    print(f"Value PCA Mode SMSE: {smse}")

    # Calculate correlations between prediction and value components
    print("Calculating component correlations.")
    correlations = np.zeros((pca_components, pca_components))
    
    for i in range(pca_components):  # True value components
        for j in range(pca_components):  # Prediction components
            corr, _ = pearsonr(y_test[:, i], predictions[:, j])
            correlations[i, j] = abs(corr)  # Absolute correlation
    
    print("Correlation matrix (rows=true, cols=prediction):")
    print(correlations)
    
    # Find the best matches for components based on correlations
    used_pred_components = set()
    component_pairs = []
    
    # Sort all correlations by strongest first
    correlation_list = []
    for i in range(pca_components):
        for j in range(pca_components):
            correlation_list.append((correlations[i, j], i, j))
    
    correlation_list.sort(reverse=True)
    
    # Pick next strongest correlation that doesn't conflict
    for corr_val, true_comp, pred_comp in correlation_list:
        if true_comp not in [pair[1] for pair in component_pairs] and pred_comp not in used_pred_components:
            component_pairs.append((corr_val, true_comp, pred_comp))
            used_pred_components.add(pred_comp)
            print(f"Matched True Component {true_comp} with Prediction Component {pred_comp} (r={corr_val:.4f})")
    
    # Create plots and detect outliers using the best component matches
    for corr_val, true_comp, pred_comp in component_pairs:
        component_col_name = f"Val-PCA-{pca_components}D - True PC{true_comp+1} vs Pred PC{pred_comp+1} (r={corr_val:.3f})"
        
        plot_graph(y_test[:, true_comp], predictions[:, pred_comp], smse, 
                    component_col_name, file_num, show_bounds, std_multiplier, frac, use_pca)

        # Create separate outlier file for each component pair
        detect_outliers(y_test[:, true_comp], predictions[:, pred_comp], x_test, 
                        complex_output_file, percent_file, component_col_name, std_multiplier, frac, mode, use_pca=True)


# This trains on and predicts values for one specific column, no PCA.
def no_pca(do_kfold, x, y, file_num, col_name, show_bounds, std_multiplier, frac, complex_output_file, percent_file, mode):
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
            detect_outliers(y_test, predictions, x_test, complex_output_file, percent_file, f"{col_name} - Fold {fold}", std_multiplier, frac, mode)

            # Store for the combined graph
            all_y_true.extend(y_test)
            all_y_pred.extend(predictions)
            all_sequences.extend(x_test)

            fold += 1
        
        else:
            # Graph and outliers for single run and end loop
            plot_graph(y_test, predictions, smse, f"{col_name}", file_num, show_bounds, std_multiplier, frac)
            detect_outliers(y_test, predictions, x_test, complex_output_file, percent_file, f"{col_name}", std_multiplier, frac, mode)

            return

    # Final combined graph after all folds
    total_smse = mean_squared_error(all_y_true, all_y_pred, squared=False)
    plot_graph(np.array(all_y_true), np.array(all_y_pred), total_smse, f"{col_name} - Combined 5-Fold", file_num, show_bounds, std_multiplier, frac)

    # Final outlier after all folds
    detect_outliers(all_y_true, all_y_pred, all_sequences, complex_output_file, percent_file, f"{col_name} - Combined", std_multiplier, frac, mode='both')


# This is the program's main function.
def main():
    # Configure variables per session
    file_num = 0  # Starts at 0
    max_seqs = 157455  # Total seqs: 157455
    balance = False
    use_pca = True
    pca_on_sequences = False
    do_kfold = True

    # Not often changed, but here for convenience
    pca_components = 10
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
    
    # Get values file and seq file
    values_file = sorted(columns_dir.glob("*.txt"))[file_num]
    col_name = values_file.name[:-4]
    seq_file = cwd / 'seqs.fa'
        
    # Prepare files for detect_outliers() function
    complex_output_file = cwd / 'outliers.txt'
    percent_file = cwd / 'outlier_percents.txt'

    if use_pca:
        print("PCA enabled - skipping folds and using full dataset.")

        # Load sequences and value files
        x, _ = load_data(seq_file, values_file, max_seqs=max_seqs, balance=balance)
        y_raw = load_all_column_targets(columns_dir, values_file, max_seqs=max_seqs, balance=balance)
        print(f"Loaded {len(x)} sequences and {len(y_raw)} value rows")

        if pca_on_sequences:
            print("Applying PCA to sequences (transposed mode)")
            pca_sequences(x, y_raw, file_num, pca_components, show_bounds, std_multiplier, frac, use_pca, complex_output_file, percent_file, mode)
        else:
            print("Applying PCA to values (original mode)")
            pca_values(x, y_raw, file_num, pca_components, show_bounds, std_multiplier, frac, use_pca, complex_output_file, percent_file, mode)

    else:
        # Load sequences and value files
        x, y = load_data(seq_file, values_file, max_seqs=max_seqs, balance=balance)
        print(f"Loaded {len(x)} sequences and {len(y)} values.")

        # No PCA use, train on one column's values only
        no_pca(do_kfold, x, y, file_num, col_name, show_bounds, std_multiplier, frac, complex_output_file, percent_file, mode)

    print("Program is finished")


# Start the program.
if __name__ == '__main__':
    main()
