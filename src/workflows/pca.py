from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.graphs.dot_plot import plot_graph
from src.model import build_model
from src.outliers import detect_outliers


# This applies PCA to values and predicts values.
def pca_values(
        x, y_raw, pca_components, show_bounds, 
        std_multiplier, frac, output_file, pct_file, mode, do_pca
):
    print(f"Loaded {x.shape[0]} sequences and {y_raw.shape[0]} value rows")
    
    # Scale PCA components for multi-output learning
    scaler_pca = StandardScaler()
    y_raw_scaled = scaler_pca.fit_transform(y_raw)

    # Use entire dataset to create PCA components
    pca = PCA(n_components=pca_components)
    # Scale before fit transform
    y_all_scaled = pca.fit_transform(y_raw)

    # Save each PCA component to 'components' folder
    components_dir = Path.cwd() / 'outputs' / 'components'
    for i in range(pca_components):
        comp_file = components_dir / f'component {i}.txt'
        np.savetxt(comp_file, y_all_scaled[:, i], fmt='%.6f')
    print(f"Saved {pca_components} PCA component files to {components_dir}")
    
    # Loop over each component, train and evaluate a single-output model
    for i in range(pca_components):
        print(f"\nTraining for PCA Component {i}")
        y_component = y_all_scaled[:, i]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y_component, test_size=0.2, random_state=42
        )

        # Build, train, and test model
        model = build_model((x.shape[1], x.shape[2]), 1)  # (500, 4)
        model.fit(
            x_train, y_train, epochs=10, batch_size=32, 
            validation_data=(x_test, y_test), verbose=0
        )
        
        predictions = model.predict(x_test).flatten()

        # Transform predictions back to original scale (original units)
        #predictions = scaler_pca.inverse_transform(predictions)
        #y_test = scaler_pca.inverse_transform(y_test)

        # Calculate metrics
        smse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Component {i} SMSE: {smse}")

        # Save results
        print(f"Creating graph and outlier files...")
        col_name = f"Val-PCA-Component-{i+1}"
        plot_graph(
            y_test, predictions, col_name, smse, show_bounds, 
            std_multiplier, frac, do_pca
        )
        detect_outliers(
            x_test, y_test, predictions, output_file, pct_file, 
            col_name, std_multiplier, frac, mode, do_pca
        )