"""
Plot comparison of Our_method vs FedAvg experiment results.
Extracts per-round average test accuracy from log files and plots comparison curves.
"""

import re
import os
import matplotlib.pyplot as plt
import numpy as np

OUR_METHOD_LABEL = "Our_method"

def parse_log_file(log_path):
    """
    Parse log file and extract per-round test accuracy.

    Returns:
        dict: {round_num: {'Art': acc, 'Clipart': acc, 'Product': acc, 'Real_World': acc, 'average': acc}}
    """
    results = {}

    # Pattern: Server: Round X Test Accuracy - Art: 0.6708, Clipart: 0.5588, Product: 0.7492, Real_World: 0.7714, average: 0.6876
    pattern = r'Round (\d+) Test Accuracy - Art: ([\d.]+), Clipart: ([\d.]+), Product: ([\d.]+), Real_World: ([\d.]+), average: ([\d.]+)'

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                round_num = int(match.group(1))
                results[round_num] = {
                    'Art': float(match.group(2)),
                    'Clipart': float(match.group(3)),
                    'Product': float(match.group(4)),
                    'Real_World': float(match.group(5)),
                    'average': float(match.group(6))
                }

    return results


def plot_comparison(exp_dir, output_path=None):
    """
    Plot comparison of all experiment logs in the directory.
    """
    # Find all log files
    log_files = []
    for f in os.listdir(exp_dir):
        if f.endswith('_exp_print.log'):
            log_files.append(os.path.join(exp_dir, f))

    if not log_files:
        print(f"No log files found in {exp_dir}")
        return

    print(f"Found {len(log_files)} log files:")
    for f in log_files:
        print(f"  - {os.path.basename(f)}")

    # Parse all log files
    all_results = {}
    for log_path in log_files:
        name = os.path.basename(log_path).replace('_exp_print.log', '')
        # Extract method name (ggeur or fedavg)
        if 'ggeur' in name.lower():
            label = OUR_METHOD_LABEL
        elif 'fedavg' in name.lower():
            label = 'FedAvg'
        else:
            label = name

        results = parse_log_file(log_path)
        if results:
            all_results[label] = results
            print(f"  Parsed {len(results)} rounds from {label}")

    if not all_results:
        print("No valid results found")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    domains = ['Art', 'Clipart', 'Product', 'Real_World', 'average']
    colors = {OUR_METHOD_LABEL: '#2196F3', 'FedAvg': '#FF5722'}
    markers = {OUR_METHOD_LABEL: 'o', 'FedAvg': 's'}

    for idx, domain in enumerate(domains):
        ax = axes[idx]

        for method, results in all_results.items():
            rounds = sorted(results.keys())
            accuracies = [results[r][domain] * 100 for r in rounds]  # Convert to percentage

            color = colors.get(method, 'gray')
            marker = markers.get(method, 'x')

            ax.plot(rounds, accuracies,
                   label=method,
                   color=color,
                   marker=marker,
                   markersize=4,
                   linewidth=1.5,
                   alpha=0.8)

        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title(f'{domain}' if domain != 'average' else 'Average', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide the 6th subplot (we only have 5 domains)
    axes[5].axis('off')

    # Add summary statistics in the 6th subplot area
    ax_text = axes[5]
    summary_text = "Final Results:\n" + "-" * 30 + "\n"

    for method, results in all_results.items():
        max_round = max(results.keys())
        final_avg = results[max_round]['average'] * 100
        best_avg = max(r['average'] for r in results.values()) * 100
        summary_text += f"\n{method}:\n"
        summary_text += f"  Final Avg: {final_avg:.2f}%\n"
        summary_text += f"  Best Avg:  {best_avg:.2f}%\n"

        # Per-domain final results
        for domain in ['Art', 'Clipart', 'Product', 'Real_World']:
            acc = results[max_round][domain] * 100
            summary_text += f"  {domain}: {acc:.2f}%\n"

    ax_text.text(0.1, 0.9, summary_text,
                transform=ax_text.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{OUR_METHOD_LABEL} vs FedAvg - Office-Home LDS (α=0.1)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = os.path.join(exp_dir, 'comparison_plot.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also show the plot
    plt.show()


def plot_average_only(exp_dir, output_path=None):
    """
    Plot only the average accuracy comparison (simpler plot).
    """
    # Find all log files
    log_files = []
    for f in os.listdir(exp_dir):
        if f.endswith('_exp_print.log'):
            log_files.append(os.path.join(exp_dir, f))

    # Parse all log files
    all_results = {}
    for log_path in log_files:
        name = os.path.basename(log_path).replace('_exp_print.log', '')
        if 'ggeur' in name.lower():
            label = OUR_METHOD_LABEL
        elif 'fedavg' in name.lower():
            label = 'FedAvg'
        else:
            label = name

        results = parse_log_file(log_path)
        if results:
            all_results[label] = results

    # Create single plot
    plt.figure(figsize=(10, 6))

    colors = {OUR_METHOD_LABEL: '#2196F3', 'FedAvg': '#FF5722'}
    markers = {OUR_METHOD_LABEL: 'o', 'FedAvg': 's'}

    for method, results in all_results.items():
        rounds = sorted(results.keys())
        accuracies = [results[r]['average'] * 100 for r in rounds]

        color = colors.get(method, 'gray')
        marker = markers.get(method, 'x')

        final_acc = accuracies[-1]
        best_acc = max(accuracies)

        plt.plot(rounds, accuracies,
                label=f'{method} (Final: {final_acc:.2f}%, Best: {best_acc:.2f}%)',
                color=color,
                marker=marker,
                markersize=5,
                linewidth=2,
                alpha=0.8)

    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Average Accuracy (%)', fontsize=12)
    plt.title(f'{OUR_METHOD_LABEL} vs FedAvg - Office-Home LDS (α=0.1)\nAverage Test Accuracy per Round',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=50)

    # Save figure
    if output_path is None:
        output_path = os.path.join(exp_dir, 'average_comparison.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    exp_dir = r'exp/GGEUR_Clip'

    print("=" * 50)
    print(f"Plotting {OUR_METHOD_LABEL} vs FedAvg Comparison")
    print("=" * 50)

    # Plot detailed comparison (all domains)
    plot_comparison(exp_dir)

    # Plot average only (simpler)
    plot_average_only(exp_dir)
