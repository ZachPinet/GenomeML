import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Configuration variables
START_LINE = 3  # Line number where data starts (1-indexed)
TRACKED_LINE = 3     # Line number to track and highlight (1-indexed)

# Files to analyze (1 or 2 files only)
RUN_DATA_FILES = [
    'run_data1.txt',
    'run_data2.txt',
]


# Read the data file from specified starting line to end of file
def read_run_data(file_path, start_line):
    titles = []
    bic_scores = []
    correlations = []
    smse_values = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Convert to 0-indexed
    start_idx = start_line - 1
    
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Split into title, BIC, correlation, and SMSE
        parts = line.split('\t')
        if len(parts) >= 4:
            titles.append(parts[0])
            bic_scores.append(float(parts[1]))
            correlations.append(float(parts[2]))
            smse_values.append(float(parts[3]))
    
    return titles, bic_scores, correlations, smse_values

# Get data from a specified line for tracking
def get_tracked_data(file_path, track_line):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if track_line > len(lines):
        return None
    
    line = lines[track_line - 1].strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None
    
    parts = line.split('\t')
    if len(parts) >= 4:
        return parts[0], float(parts[1]), float(parts[2]), float(parts[3])
    
    return None


# Create a boxplot from one or two datasets
def create_boxplot(
        data_list, tracked_data_list, metric_name, 
        ylabel, filename, lower_is_better=True
):
    num_datasets = len(data_list)
    colors = ['lightcoral', 'lightblue']
    
    # Create file labels (remove .txt extension)
    file_labels = [name[:-4] for name in RUN_DATA_FILES[:num_datasets]]
    
    # Set up positions to avoid overlap between key, box, and labels
    if num_datasets == 1:
        box_positions = [3]
        text_positions = [3.5]
        fig_width = 9
        xlim = (0.5, 6)
    else:
        box_positions = [2, 4]
        text_positions = [2.5, 4.5]
        fig_width = 12
        xlim = (0.5, 7)
    
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    # Create boxplots
    bp = ax.boxplot(
        data_list, positions=box_positions, patch_artist=True, 
        widths=0.4, medianprops=dict(color='black', linewidth=2)
    )
    
    # Color the boxes
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)
    
    # Add tracked points if provided
    for i, tracked_data in enumerate(tracked_data_list):
        if tracked_data:
            title, tracked_value = tracked_data
            ax.plot(
                box_positions[i], tracked_value, 'ro', markersize=10, 
                    markerfacecolor='red', markeredgecolor='darkred', 
                    markeredgewidth=2, zorder=10
            )
    
    # Add min, max, median at corresponding y-levels
    for i, data in enumerate(data_list):
        min_val = np.min(data)
        max_val = np.max(data)
        median_val = np.median(data)
        
        text_x = text_positions[i]
        
        # Place the text
        ax.text(
            text_x, max_val, f'Max: {max_val:.4f}', 
            verticalalignment='center', horizontalalignment='left',
            bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white', 
                alpha=0.8, edgecolor='black'
            ), 
            fontsize=10, fontweight='bold'
        )
        
        ax.text(
            text_x, median_val, f'Median: {median_val:.4f}', 
            verticalalignment='center', horizontalalignment='left',
            bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white', 
                alpha=0.8, edgecolor='black'
            ), 
            fontsize=10, fontweight='bold'
        )
        
        ax.text(
            text_x, min_val, f'Min: {min_val:.4f}', 
            verticalalignment='center', horizontalalignment='left',
            bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white', 
                alpha=0.8, edgecolor='black'
            ), 
            fontsize=10, fontweight='bold'
        )
    
    # Create keys in the corners
    for i, data in enumerate(data_list):
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Add tracked info if available
        tracked_info = ""
        if tracked_data_list[i]:
            title, tracked_value = tracked_data_list[i]
            tracked_info = (
                f"\nTracked Line: {TRACKED_LINE}\n"
                "Value: {tracked_value:.4f}\n({title})"
            )
        
        stats_text = (
            f'{file_labels[i]}:\n'
            f'Count: {len(data)}\n'
            f'Mean: {mean_val:.4f}\n'
            f'Std: {std_val:.4f}'
            f'{tracked_info}'
        )
        
        # Position legend boxes in corners
        if num_datasets == 1:
            # Top left for single dataset
            x_pos = 0.02
            y_pos = 0.98
        else:
            # Top left for first dataset, top right for second
            x_pos = 0.02 if i == 0 else 0.98
            y_pos = 0.98
            
        ax.text(
            x_pos, y_pos, stats_text, 
            transform=ax.transAxes, 
            verticalalignment='top', 
            horizontalalignment='left' if i == 0 else 'right',
            bbox=dict(
                boxstyle='round', facecolor=colors[i], 
                alpha=0.8, edgecolor='black'
            ),
            fontsize=11
        )
    
    # Formatting
    better_text = "Lower is Better" if lower_is_better else "Higher is Better"
    title = f'{metric_name} Comparison\n({better_text})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis labels and limits
    ax.set_xlim(xlim)
    ax.set_xticks(box_positions)
    ax.set_xticklabels(file_labels, fontsize=14, fontweight='bold')
    
    # Save results to outputs folder
    save_dir = Path.cwd() / 'outputs' / 'graphs'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Boxplot saved: {save_path}")
    plt.close()


# Main function to create boxplots for run data
def main(): 
    # Validate configuration
    if len(RUN_DATA_FILES) < 1 or len(RUN_DATA_FILES) > 2:
        print("Error: Must specify 1 or 2 run data files!")
        return
    
    # Check if files exist
    run_data_dir = Path.cwd() / 'outputs' / 'run_data'
    file_paths = []
    
    for filename in RUN_DATA_FILES:
        file_path = run_data_dir / filename
        if not file_path.exists():
            print(f"Error: {file_path} not found!")
            return
        file_paths.append(file_path)
    
    # Read data from all files
    all_titles = []
    all_bic_scores = []
    all_correlations = []
    all_smse_values = []
    all_tracked_info = []
    
    for i, file_path in enumerate(file_paths):
        print(f"Reading from {RUN_DATA_FILES[i]} (start line {START_LINE})...")
        titles, bic_scores, correlations, smse_values = read_run_data(
            file_path, START_LINE
        )
        
        if not bic_scores:
            print(f"No data found in {RUN_DATA_FILES[i]}!")
            return
        
        all_titles.append(titles)
        all_bic_scores.append(bic_scores)
        all_correlations.append(correlations)
        all_smse_values.append(smse_values)
        
        # Get tracked data
        tracked_info = get_tracked_data(file_path, TRACKED_LINE)
        all_tracked_info.append(tracked_info)
    
    # Create boxplots
    print("\nCreating boxplots...")
    
    # Prepare tracked data for each metric
    bic_tracked = [
        (info[0], info[1]) if info else None for info in all_tracked_info
    ]
    corr_tracked = [
        (info[0], info[2]) if info else None for info in all_tracked_info
    ]
    smse_tracked = [
        (info[0], info[3]) if info else None for info in all_tracked_info
    ]
    
    # Create the three boxplots
    create_boxplot(
        all_bic_scores, bic_tracked,
        'BIC Scores', 'BIC Score', 'BIC_boxplot.png', 
        lower_is_better=True
    )
    
    create_boxplot(
        all_correlations, corr_tracked,
        'Correlation Coefficients', 'Correlation', 'Correlation_boxplot.png', 
        lower_is_better=False
    )
    
    create_boxplot(
        all_smse_values, smse_tracked,
        'SMSE Values', 'SMSE', 'SMSE_boxplot.png', 
        lower_is_better=True
    )
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Show tracked data if available
    for i, filename in enumerate(RUN_DATA_FILES):
        if all_tracked_info[i]:
            tracked_info = all_tracked_info[i]
            print(f"\n  Tracked data (line {TRACKED_LINE}): {tracked_info[0]}")
            print(f"    BIC: {tracked_info[1]:.4f}")
            print(f"    Correlation: {tracked_info[2]:.4f}")
            print(f"    SMSE: {tracked_info[3]:.4f}")
    
    print(f"\nAll boxplots saved to outputs/graphs/")
    print(f"Files created:")
    print(f"  - BIC_boxplot.png")
    print(f"  - Correlation_boxplot.png")
    print(f"  - SMSE_boxplot.png")


if __name__ == '__main__':
    main()