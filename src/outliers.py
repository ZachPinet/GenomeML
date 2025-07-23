# outlier_pie_chart.py
import matplotlib.pyplot as plt
from collections import Counter


# This returns the # of times each sequence has been an outlier.
def load_outlier_counts(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    outlier_counts = [line.strip().split('\t')[1] for line in lines]
    return outlier_counts


# This creates a pie chart to display the outlier count data.
def make_pie_chart(outlier_counts, total_seq_count):
    counter = Counter(outlier_counts)
    total_outlier_count = 0
    for count in counter:
        total_outlier_count += counter[count]

    # Calculate "0 times" sequences
    zero_times = total_seq_count - total_outlier_count

    # Prepare data for the pie chart
    labels = []
    sizes = []

    if zero_times > 0:
        sizes.append(zero_times)
        labels.append(f'0 Times ({zero_times} sequences)')

    for times, count in sorted(counter.items(), key=lambda x: int(x[0])):
        sizes.append(count)
        time_label = f'{times} Time' if times == '1' else f'{times} Times'
        percent = (count / total_seq_count) * 100
        labels.append(f'{time_label} ({count} sequences, {percent:.1f}%)')

    # Plotting
    fig, ax = plt.subplots()
    wedges, _ = ax.pie(sizes, startangle=140)
    ax.legend(wedges, labels, title="Outlier Frequency", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title('Distribution of Outlier Sequences')
    plt.tight_layout()
    plt.show()


# This is the main function.
def main():
    outliers_file = 'outliers.txt'
    total_seq_count = 285250
    outlier_counts = load_outlier_counts(outliers_file)
    make_pie_chart(outlier_counts, total_seq_count)


if __name__ == '__main__':
    main()
