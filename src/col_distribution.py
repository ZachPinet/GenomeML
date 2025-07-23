import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# This reads the values from the col files.
def read_values(file_path):
    with open(file_path, 'r') as file:
        values = [float(line.strip()) for line in file if line.strip()]
    return values


# This creates a boxplot and a histogram side-by-side.
def plot_distribution(values, file_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    file_base = file_name[:-4]   # omit '.txt' file extension
    fig.suptitle(f"Distribution of Values from {file_base}", fontsize=16, fontweight='bold')

    # Boxplot
    sns.boxplot(x=values, color='lightblue', ax=axes[0])
    axes[0].set_title("Boxplot of Values")
    axes[0].set_xlabel("Value")

    # Histogram
    bins = np.linspace(-10, 10, 101)
    axes[1].hist(values, bins=bins, color='lightgreen', edgecolor='black')
    axes[1].set_title("Histogram of Value Distribution")
    axes[1].set_xlabel("Value Range")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure to Downloads folder
    save_path = save_dir / f"{file_base} Distribution.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

    plt.show()


# This is main.
def main():
    columns_dir = Path.cwd() / 'columns'
    downloads_dir = Path.home() / 'Downloads'   # save imgs to downloads

    # Ensure the 'columns' folder exists
    if not columns_dir.exists():
        print(f"'columns' directory not found at {columns_dir}")
        return
    
    # Loop through all .txt files in the columns directory
    for file_path in sorted(columns_dir.glob("*.txt")):
        values = read_values(file_path)
        plot_distribution(values, file_path.name, downloads_dir)


if __name__ == "__main__":
    main()
