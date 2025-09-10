import os
import csv
import argparse
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def find_metric_files(root_dir, target_filename="test_metrics_summary.csv"):
    """Recursively find all files named `target_filename` under root_dir."""
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        if target_filename in filenames:
            matches.append(os.path.join(dirpath, target_filename))
    return matches

def process_metrics_file(file_path):
    """Extract experiment name from parent dir and parse the metrics."""
    try:
        experiment_name = os.path.basename(os.path.dirname(file_path))
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            return {"experiment": experiment_name, **dict(zip(header, row))}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Parse test_metrics_summary.csv files and aggregate metrics.")
    parser.add_argument("--log_file", required=True, help="Root directory to search for test_metrics_summary.csv files.")
    parser.add_argument("--metrics_file", required=True, help="Path to output CSV file.")

    args = parser.parse_args()
    all_files = find_metric_files(args.log_file)

    results = []
    with ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(process_metrics_file, all_files), total=len(all_files), desc="Parsing metric files"):
            if result:
                results.append(result)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.metrics_file, index=False)
        print(f"\nMetrics summary saved to: {args.metrics_file}")
    else:
        print("No valid metrics found.")

if __name__ == "__main__":
    main()
