"""Compare Random Search and Iterated Local Search for each dataset."""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import glob
from baseline import random_search
from iterated_local_search import ils_algorithm

def compare_metrics(dataset_path, budget, repetitions=30, rs_output_dir="rs_results", ils_output_dir="ils_results"):
    rs_results = []
    ils_results = []
    
    dataset_name = os.path.basename(dataset_path).replace('.csv', '')
    print(f"\nStarting comparison for {dataset_name}...")

    # Run Random Search multiple times
    print(f"Running Random Search {repetitions} times...")
    for i in range(repetitions):
        output_file = os.path.join(rs_output_dir, f"rs_run_{dataset_name}_{i+1}.csv")
        _, rs_performance = random_search(dataset_path, budget, output_file)
        rs_results.append(rs_performance)

        if (i + 1) % 10 == 0:
            print(f"RS iteration {i+1}/{repetitions} complete.")

    # Run ILS multiple times
    print(f"Running ILS {repetitions} times...")
    for i in range(repetitions):
        output_file = os.path.join(ils_output_dir, f"ils_run_{dataset_name}_{i+1}.csv")
        _, ils_performance = ils_algorithm(dataset_path, budget, output_file)
        ils_results.append(ils_performance)

        if (i + 1) % 10 == 0:
            print(f"ILS iteration {i+1}/{repetitions} complete.")

    # Statistical analysis using Wilcoxon rank-sum test
    stat, p_value = stats.mannwhitneyu(rs_results, ils_results, alternative='two-sided')

    # Calculate statistics
    rs_best = np.min(rs_results)
    rs_median = np.median(rs_results)
    rs_mean = np.mean(rs_results)
    ils_best = np.min(ils_results)
    ils_median = np.median(ils_results)
    ils_mean = np.mean(ils_results)

    print(f"\n--- Results for {dataset_name} ---")
    print(f"Random Search - Best: {rs_best:.4f}, Mean: {rs_mean:.4f}, Median: {rs_median:.4f}")
    print(f"ILS          - Best: {ils_best:.4f}, Mean: {ils_mean:.4f}, Median: {ils_median:.4f}")
    print(f"Wilcoxon rank-sum test p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"Significant difference detected (p < 0.05)")
    else:
        print(f"No significant difference (p >= 0.05)")

    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.boxplot([rs_results, ils_results], labels=['Random Search', 'ILS'])
    plt.ylabel('Performance')
    plt.title(f'Performance Comparison: {dataset_name}')
    plt.grid(axis='y', alpha=0.3)
    vis_file = os.path.join('visualisations', f'comparison_boxplot_{dataset_name}.png')
    plt.savefig(vis_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'dataset': dataset_name,
        'rs_best': rs_best,
        'rs_mean': rs_mean,
        'rs_median': rs_median,
        'ils_best': ils_best,
        'ils_mean': ils_mean,
        'ils_median': ils_median,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def main():
    datasets_folder = "datasets"
    rs_output_dir = "rs_results"
    ils_output_dir = "ils_results"
    vis_output_dir = "visualisations"
    summary_output_dir = "results"
    budget = 100
    repetitions = 30
    
    # Create output directories
    os.makedirs(rs_output_dir, exist_ok=True)
    os.makedirs(ils_output_dir, exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)
    os.makedirs(summary_output_dir, exist_ok=True)
    
    # Get all dataset files
    dataset_files = sorted(glob.glob(os.path.join(datasets_folder, "*.csv")))
    
    if not dataset_files:
        print(f"No datasets found in {datasets_folder}")
        return
    
    results_summary = []
    
    for dataset_file in dataset_files:
        dataset_name = os.path.basename(dataset_file).replace('.csv', '')
        
        try:
            results = compare_metrics(dataset_file, budget, repetitions, rs_output_dir, ils_output_dir)
            results_summary.append(results)
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
    
    # Print summary
    if results_summary:
        print("\n" + "-"*100)
        print("Results Summary:")
        print("-"*100)
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_file = os.path.join(summary_output_dir, 'comparison_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to {summary_file}")

if __name__ == "__main__":
    main()



