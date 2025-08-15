from pathlib import Path

import matplotlib.pyplot as plt


# Creates a line chart from given values and labels
def create_line_chart(values, labels, ylabel="Values", title="Line Chart"):
    plt.figure(figsize=(10, 6))
    plt.plot(labels, values, color='skyblue', marker='o', linewidth=2)
    plt.xlabel("Labels", fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_dir = Path.cwd() / 'outputs' / 'graphs'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{title}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Line graph saved: {save_path}")

# Example usage:
if __name__ == "__main__":
    values = [1, 4, 3, 2, 5]
    labels = ['A', 'B', 'C', 'D', 'E']
    create_line_chart(values, labels, ylabel="Values", title="Chart Title")