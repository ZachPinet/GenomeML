from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.smoothers_lowess import lowess


# This makes a heatmap scatterplot with the data.
def plot_graph(
        y_true, y_pred, col_name, smse, show_bounds, 
        std_multiplier, frac, do_pca=False
):
    plt.figure(figsize=(10, 6))

    # Calculate Pearson correlation for the graph title
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    xy = np.vstack([y_true, y_pred])
    density = gaussian_kde(xy)(xy)
    plt.scatter(
        y_true, y_pred, c=density, cmap="RdYlGn_r", 
        edgecolors="none", alpha=0.6, s=4
    )
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

        plt.plot(
            x_loess, upper, color='gray', linewidth=0.5, 
            linestyle='--', label=f'+{std_multiplier}σ'
        )
        plt.plot(
            x_loess, lower, color='gray', linewidth=0.5, 
            linestyle='--', label=f'-{std_multiplier}σ'
        )

    # Labels
    xlabel = 'True Values (PCA)' if do_pca else 'True Values'
    ylabel = 'Predicted Values (PCA)' if do_pca else 'Predicted Values'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{col_name} (r={corr:.3f}) (SMSE: {smse:.4f})', fontsize=11)
    plt.legend()

    # Save the figure to graphs folder
    graphs_dir = Path.cwd() / 'outputs' / 'graphs'
    save_name = f"{col_name}.png"
    save_path = graphs_dir / save_name
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.close()