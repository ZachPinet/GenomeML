import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def read_bic_data(file_path, start_line, end_line=None):
    """
    Read BIC data from specified line range
    
    Args:
        file_path: Path to bic_scores.txt file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed), None for end of file
    
    Returns:
        tuple: (scenario_names, bic_scores, correlations, smse_values)
    """
    scenario_names = []
    bic_scores = []
    correlations = []
    smse_values = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Convert to 0-indexed and handle end_line
    start_idx = start_line - 1
    end_idx = len(lines) if end_line is None else end_line
    
    for i in range(start_idx, min(end_idx, len(lines))):
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        parts = line.split('\t')
        if len(parts) >= 5:
            scenario_names.append(parts[0])
            bic_scores.append(float(parts[1]))
            correlations.append(float(parts[2]))
            smse_values.append(float(parts[3]))
    
    return scenario_names, bic_scores, correlations, smse_values

def get_tracked_data(file_path, track_line):
    """
    Get data from a specific line for tracking
    
    Args:
        file_path: Path to bic_scores.txt file
        track_line: Line number to track (1-indexed)
    
    Returns:
        tuple: (scenario_name, bic_score, correlation, smse_value) or None if not found
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if track_line > len(lines):
        return None
    
    line = lines[track_line - 1].strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None
    
    parts = line.split('\t')
    if len(parts) >= 5:
        return parts[0], float(parts[1]), float(parts[2]), float(parts[3])
    
    return None

def create_multi_boxplot(all_data, metric_name, title, ylabel, filename, colors=None, tracked_data_list=None):
    """
    Create a boxplot with multiple datasets side by side
    
    Args:
        all_data: List of data lists for each dataset
        metric_name: Name of the metric being plotted
        title: Plot title
        ylabel: Y-axis label
        filename: Output filename
        colors: List of colors for each dataset
        tracked_data_list: List of tuples (scenario_name, value) for each dataset, or None
    """
    if colors is None:
        colors = ['lightcoral', 'lightgreen', 'lightblue', 'lightyellow', 'lightpink']
    
    fig, ax = plt.subplots(figsize=(max(12, 4 * len(all_data)), 8))
    
    # Create boxplots
    positions = range(1, len(all_data) + 1)
    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, 
                    tick_labels=[f'Dataset {i+1}' for i in range(len(all_data))])
    
    # Color the boxes
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.7)
    
    # Add tracked points if provided
    if tracked_data_list:
        for i, tracked_data in enumerate(tracked_data_list):
            if tracked_data:
                scenario_name, tracked_value = tracked_data
                ax.plot(i + 1, tracked_value, 'ro', markersize=8, markerfacecolor='red', 
                        markeredgecolor='darkred', markeredgewidth=2, zorder=10)
    
    # Create statistics text boxes for each dataset
    legend_elements = []
    y_positions = np.linspace(0.95, 0.05, len(all_data))
    
    for i, data in enumerate(all_data):
        min_val = np.min(data)
        max_val = np.max(data)
        median_val = np.median(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Add tracked info if available
        tracked_info = ""
        if tracked_data_list and tracked_data_list[i]:
            scenario_name, tracked_value = tracked_data_list[i]
            tracked_info = f'\nTracked: {tracked_value:.4f}\n({scenario_name})'
        
        stats_text = (f'Dataset {i+1}:\n'
                     f'Count: {len(data)}\n'
                     f'Min: {min_val:.4f}\n'
                     f'Max: {max_val:.4f}\n'
                     f'Median: {median_val:.4f}\n'
                     f'Mean: {mean_val:.4f}\n'
                     f'Std: {std_val:.4f}'
                     f'{tracked_info}')
        
        # Position legend boxes vertically
        ax.text(0.02, y_positions[i], stats_text, transform=ax.transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor=colors[i % len(colors)], 
                         alpha=0.8, edgecolor='black'),
                fontsize=9)
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Save to Downloads folder
    save_dir = Path.home() / 'Downloads'
    save_path = save_dir / filename
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Boxplot saved: {save_path}")
    plt.close()

def main():
    """Main function to create all three boxplots"""
    
    # Configuration - Multiple files for comparison
    bic_files = [
        Path.cwd() / 'bic_scores.txt',
        # Uncomment lines below to add more files for comparison:
        # Path.cwd() / 'bic_scores2.txt',
        # Path.cwd() / 'bic_scores3.txt',
    ]
    dataset_labels = [
        'Dataset 1',
        # Uncomment and modify labels for additional datasets:
        # 'Dataset 2',
        # 'Dataset 3',
    ]
    
    start_line = 7  # Line 7 in your file (1-indexed)
    end_line = None  # None means read to end of file
    
    # TRACKING CONFIGURATION
    track_lines = [7]  # List of line numbers to track for each file (None to disable tracking)
    # For multiple files, use: track_lines = [7, 7, 7]  # One for each file
    
    # Validate configuration
    if len(bic_files) != len(dataset_labels):
        print("Error: Number of files must match number of dataset labels!")
        return
    
    if track_lines and len(track_lines) != len(bic_files):
        print("Error: Number of track_lines must match number of files!")
        return
    
    # Check if files exist
    for i, bic_file in enumerate(bic_files):
        if not bic_file.exists():
            print(f"Error: {bic_file} not found!")
            return
    
    # Read data from all files
    all_scenario_names = []
    all_bic_scores = []
    all_correlations = []
    all_smse_values = []
    all_tracked_info = []
    
    for i, bic_file in enumerate(bic_files):
        print(f"Reading BIC data from {dataset_labels[i]} (line {start_line} to end)...")
        scenario_names, bic_scores, correlations, smse_values = read_bic_data(
            bic_file, start_line, end_line
        )
        
        if not bic_scores:
            print(f"No data found in {dataset_labels[i]}!")
            return
        
        all_scenario_names.append(scenario_names)
        all_bic_scores.append(bic_scores)
        all_correlations.append(correlations)
        all_smse_values.append(smse_values)
        
        print(f"  Loaded {len(bic_scores)} data points")
        print(f"  BIC range: {min(bic_scores):.2f} - {max(bic_scores):.2f}")
        print(f"  Correlation range: {min(correlations):.4f} - {max(correlations):.4f}")
        print(f"  SMSE range: {min(smse_values):.4f} - {max(smse_values):.4f}")
        
        # Get tracked data if tracking is enabled
        tracked_info = None
        if track_lines and track_lines[i]:
            tracked_info = get_tracked_data(bic_file, track_lines[i])
            if tracked_info:
                print(f"  Tracking line {track_lines[i]}: {tracked_info[0]}")
                print(f"    BIC: {tracked_info[1]:.4f}")
                print(f"    Correlation: {tracked_info[2]:.4f}")
                print(f"    SMSE: {tracked_info[3]:.4f}")
            else:
                print(f"  Warning: Could not find data for line {track_lines[i]}")
        
        all_tracked_info.append(tracked_info)
    
    # Create comparison boxplots
    print("\nCreating comparison boxplots...")
    
    # Prepare tracked data for each metric
    bic_tracked = [(info[0], info[1]) if info else None for info in all_tracked_info]
    corr_tracked = [(info[0], info[2]) if info else None for info in all_tracked_info]
    smse_tracked = [(info[0], info[3]) if info else None for info in all_tracked_info]
    
    # BIC Scores comparison boxplot
    track_info_str = "" if not any(track_lines) else f" - Tracking Lines {track_lines}"
    create_multi_boxplot(
        all_bic_scores, 
        'BIC',
        f'BIC Scores Comparison\n(Lower is Better){track_info_str}', 
        'BIC Score', 
        'BIC_comparison_boxplot.png',
        colors=['lightcoral', 'lightgreen', 'lightblue', 'lightyellow', 'lightpink'],
        tracked_data_list=bic_tracked
    )
    
    # Correlations comparison boxplot
    create_multi_boxplot(
        all_correlations, 
        'Correlation',
        f'Correlation Coefficients Comparison\n(Higher is Better){track_info_str}', 
        'Correlation', 
        'Correlation_comparison_boxplot.png',
        colors=['lightgreen', 'lightcoral', 'lightblue', 'lightyellow', 'lightpink'],
        tracked_data_list=corr_tracked
    )
    
    # SMSE comparison boxplot
    create_multi_boxplot(
        all_smse_values, 
        'SMSE',
        f'SMSE Comparison\n(Lower is Better){track_info_str}', 
        'SMSE', 
        'SMSE_comparison_boxplot.png',
        colors=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink'],
        tracked_data_list=smse_tracked
    )
    
    # Summary statistics for all datasets
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - ALL DATASETS")
    print("="*80)
    
    for i, label in enumerate(dataset_labels):
        print(f"\n{label}:")
        print(f"{'Metric':<12} {'Min':<10} {'Max':<10} {'Median':<10} {'Mean':<10} {'Std':<10}")
        print("-" * 70)
        print(f"{'BIC':<12} {min(all_bic_scores[i]):<10.2f} {max(all_bic_scores[i]):<10.2f} {np.median(all_bic_scores[i]):<10.2f} {np.mean(all_bic_scores[i]):<10.2f} {np.std(all_bic_scores[i]):<10.2f}")
        print(f"{'Correlation':<12} {min(all_correlations[i]):<10.4f} {max(all_correlations[i]):<10.4f} {np.median(all_correlations[i]):<10.4f} {np.mean(all_correlations[i]):<10.4f} {np.std(all_correlations[i]):<10.4f}")
        print(f"{'SMSE':<12} {min(all_smse_values[i]):<10.4f} {max(all_smse_values[i]):<10.4f} {np.median(all_smse_values[i]):<10.4f} {np.mean(all_smse_values[i]):<10.4f} {np.std(all_smse_values[i]):<10.4f}")
        
        # Show tracked data position in rankings
        if all_tracked_info[i]:
            tracked_info = all_tracked_info[i]
            print(f"\n  === TRACKED DATA RANKINGS ===")
            
            # BIC ranking (lower is better)
            bic_rank = sum(1 for x in all_bic_scores[i] if x < tracked_info[1]) + 1
            print(f"  BIC Rank: {bic_rank}/{len(all_bic_scores[i])} (lower is better)")
            
            # Correlation ranking (higher is better) 
            corr_rank = sum(1 for x in all_correlations[i] if x > tracked_info[2]) + 1
            print(f"  Correlation Rank: {corr_rank}/{len(all_correlations[i])} (higher is better)")
            
            # SMSE ranking (lower is better)
            smse_rank = sum(1 for x in all_smse_values[i] if x < tracked_info[3]) + 1
            print(f"  SMSE Rank: {smse_rank}/{len(all_smse_values[i])} (lower is better)")
        
        # Find best models for this dataset
        print(f"\n  === BEST MODELS ===")
        
        # Best BIC (lowest)
        best_bic_idx = np.argmin(all_bic_scores[i])
        print(f"  Best BIC: {all_scenario_names[i][best_bic_idx]} (BIC: {all_bic_scores[i][best_bic_idx]:.2f})")
        
        # Best Correlation (highest)
        best_corr_idx = np.argmax(all_correlations[i])
        print(f"  Best Correlation: {all_scenario_names[i][best_corr_idx]} (r: {all_correlations[i][best_corr_idx]:.4f})")
        
        # Best SMSE (lowest)
        best_smse_idx = np.argmin(all_smse_values[i])
        print(f"  Best SMSE: {all_scenario_names[i][best_smse_idx]} (SMSE: {all_smse_values[i][best_smse_idx]:.4f})")
    
    print(f"\nAll comparison boxplots saved to Downloads folder!")
    print(f"Files created:")
    print(f"  - BIC_comparison_boxplot.png")
    print(f"  - Correlation_comparison_boxplot.png") 
    print(f"  - SMSE_comparison_boxplot.png")

if __name__ == '__main__':
    main()