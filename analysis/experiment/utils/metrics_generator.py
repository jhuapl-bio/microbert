# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from multiprocessing import Process
from itertools import combinations


class MetricsGenerator:
    def __init__(self, folder_path):
        """
        Initialize the MetricsGenerator class.

        Args:
            folder_path (str): Path to the folder containing prediction CSV files.
        """
        self.folder_path = folder_path
        self.metrics_results = {}

    def compute_metrics(self, predicted_labels, actual_labels):
        """
        Compute precision, recall, F1-score, and support for each class.

        Args:
            predicted_labels (array-like): Predicted labels.
            actual_labels (array-like): True labels.

        Returns:
            pd.DataFrame: Dataframe containing metrics for each class.
        """
        classes = np.unique(np.concatenate((actual_labels, predicted_labels)))
        precision, recall, f1, support = precision_recall_fscore_support(
            actual_labels, predicted_labels, labels=classes, zero_division=0
        )

        metrics_df = pd.DataFrame(
            {
                "Class": classes,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "Support": support,
            }
        )

        return metrics_df

    def compute_correlation(self, metrics_df):
        """
        Compute and print the correlation between Support and F1-score.

        Args:
            metrics_df (pd.DataFrame): DataFrame containing metrics including 'F1-score' and 'Support'.
        """
        if "F1-score" in metrics_df.columns and "Support" in metrics_df.columns:
            correlation = metrics_df["Support"].corr(metrics_df["F1-score"])
            print(f"Correlation between Support and F1-score: {correlation:.4f}")
        else:
            print(
                "Metrics DataFrame must contain 'F1-score' and 'Support' columns to compute correlation."
            )

    def save_confusion_matrix(self, predicted_labels, actual_labels, file_name):
        """
        Generate and save a confusion matrix as a PDF.

        Args:
            predicted_labels (array-like): Predicted labels.
            actual_labels (array-like): True labels.
            file_name (str): Name of the file to save the confusion matrix.
        """
        cm = confusion_matrix(actual_labels, predicted_labels)
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot(cmap="Blues", values_format="d")

        pdf_file_name = f"{file_name}_confusion_matrix.pdf"
        plt.savefig(pdf_file_name)
        plt.close()
        print(f"Confusion matrix saved as {pdf_file_name}")

    def save_metrics_to_csv(self, metrics_df, model_path):
        """
        Save the computed metrics to a CSV file in the specified model path.

        Args:
            metrics_df (pd.DataFrame): DataFrame containing the metrics.
            model_path (str): Path to the model directory where the CSV will be saved.
        """
        csv_file_path = os.path.join(model_path, "class_metrics.csv")
        metrics_df.to_csv(csv_file_path, index=False)
        print(f"Metrics saved to {csv_file_path}")

    def process_model(self):
        """
        Process all CSV files in the folder and compute metrics.

        Returns:
            pd.DataFrame: Combined dataframe with metrics for all files.
        """
        all_metrics = []

        for file_name in os.listdir(self.folder_path):
            if file_name.endswith("_predictions.csv"):
                file_path = os.path.join(self.folder_path, file_name)
                try:
                    results_df = pd.read_csv(file_path, index_col=0)

                    # Ensure the required columns exist
                    if (
                        "Predicted_Label" not in results_df
                        or "Actual_Label" not in results_df
                    ):
                        raise ValueError(
                            "File must contain 'Predicted_Label' and 'Actual_Label' columns."
                        )

                    predicted_labels = results_df["Predicted_Label"].values
                    actual_labels = results_df["Actual_Label"].values

                    metrics_df = self.compute_metrics(predicted_labels, actual_labels)
                    metrics_df["Rank"] = file_name.split("_")[
                        0
                    ]  # Extract a rank or identifier from the filename

                    all_metrics.append(metrics_df)

                except ValueError as e:
                    print(f"ValueError for {file_name}: {e}")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        # Combine all metrics into one DataFrame
        if all_metrics:
            combined_metrics_df = pd.concat(all_metrics, ignore_index=True)
            combined_metrics_df = combined_metrics_df[
                ["Rank", "Class", "Precision", "Recall", "F1-score", "Support"]
            ]

            return combined_metrics_df
        else:
            print("No valid prediction files found.")
            return pd.DataFrame()

    def process_multiple_models(self, directory_path):
        """
        Process multiple model directories within a parent directory.

        Args:
            directory_path (str): Path to the directory containing multiple model subdirectories.
        """

        def process_single_model(model_path):
            test_results_path = os.path.join(model_path, "test_results")
            if os.path.isdir(test_results_path):
                print(f"Processing model directory: {test_results_path}")
                self.folder_path = test_results_path
                metrics_df = self.process_model()

                if not metrics_df.empty:
                    self.save_metrics_to_csv(metrics_df, test_results_path)
            else:
                print(f"Skipped {model_path}, 'test_results' folder not found.")

        processes = []
        for model_dir in os.listdir(directory_path):
            model_path = os.path.join(directory_path, model_dir)
            if os.path.isdir(model_path):
                p = Process(target=process_single_model, args=(model_path,))
                processes.append(p)
                p.start()

        for p in processes:
            p.join()

    def compute_jaccard_similarity(self, parent_dir):
        """
        Compute Jaccard similarity between model predictions for each taxonomy level and save to CSV.

        Args:
            parent_dir (str): Path to the parent directory containing model subdirectories.
        """
        model_dirs = [
            os.path.join(parent_dir, d, "test_results")
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d, "test_results"))
        ]
        similarities = []

        for model1, model2 in combinations(model_dirs, 2):
            model1_name = os.path.basename(os.path.dirname(model1))
            model2_name = os.path.basename(os.path.dirname(model2))

            model1_files = [
                f for f in os.listdir(model1) if f.endswith("_predictions.csv")
            ]
            model2_files = [
                f for f in os.listdir(model2) if f.endswith("_predictions.csv")
            ]

            common_files = set(model1_files) & set(model2_files)

            for file in common_files:
                df1 = pd.read_csv(os.path.join(model1, file))
                df2 = pd.read_csv(os.path.join(model2, file))

                if "Predicted_Label" in df1 and "Predicted_Label" in df2:
                    set1 = set(df1["Predicted_Label"])
                    set2 = set(df2["Predicted_Label"])

                    jaccard = len(set1 & set2) / len(set1 | set2)
                    taxonomy_level = file.split("_")[
                        0
                    ]  # Extract taxonomy level from the filename

                    similarities.append(
                        {
                            "Model_1": model1_name,
                            "Model_2": model2_name,
                            "Taxonomy_Level": taxonomy_level,
                            "Jaccard_Similarity": jaccard,
                        }
                    )

        if similarities:
            similarities_df = pd.DataFrame(similarities)
            output_path = os.path.join(parent_dir, "jaccard_similarity.csv")
            similarities_df.to_csv(output_path, index=False)
            print(f"Jaccard similarity results saved to {output_path}")
        else:
            print("No common files found to compute Jaccard similarity.")


# Example usage:
if __name__ == "__main__":
    # Parent directory containing multiple model subdirectories
    parent_dir = "/home/apluser/analysis/analysis/experiment/runs/bertax/full/"
    generator = MetricsGenerator(None)
    generator.process_multiple_models(parent_dir)
    # generator.compute_jaccard_similarity(parent_dir)
