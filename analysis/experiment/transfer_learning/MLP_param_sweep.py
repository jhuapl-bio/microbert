# driver_experiments.py

import os
import copy
import argparse
from itertools import product
from train_MLP import main  # or your main training script

# from train_MLP_unconditioned import main as unconditioned_main
from analysis.experiment.utils.config import CONFIG_DIR, Config

# from transformers import AutoTokenizer
import pandas as pd

# from analysis.experiment.process.hierarchical_processor import HierarchicalDataProcessor
from analysis.experiment.utils.embeddings_utils import read_group_in_batches

# import time
import numpy as np

# from argparse import ArgumentParser

import torch

# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# from transformers import AutoTokenizer
# from tqdm import tqdm

# from analysis.experiment.process.hierarchical_processor import HierarchicalDataProcessor
from analysis.experiment.utils.train_logger import logger

# from analysis.experiment.utils.train_utils import multiclass_metrics


class HierarchicalDataset(Dataset):
    """
    Holds embeddings plus up to 3 label columns:
      superkingdom_labels[i], phylum_labels[i], genus_labels[i]
    """

    def __init__(self, embeddings, label_matrix):
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        if isinstance(label_matrix, np.ndarray):
            label_matrix = torch.from_numpy(label_matrix).long()

        self.embeddings = embeddings
        self.labels = label_matrix  # shape [N, 3]

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def run_all_experiments(
    config_template: Config, unconditioned: bool = False, merge_only: bool = False
):
    """
    Enumerates hyperparameter combos, creates a subfolder under
    <config_template.save_dir>/MLP_experiments for each experiment,
    and runs training + evaluation sequentially in that subfolder.
    Skips experiments that appear to be completed already.

    After all experiments finish, compiles the results from each experiment's
    'test_metrics_summary.csv' into a single merged CSV.
    """
    # 1. Define your hyperparameter grids:
    num_layers_list = [3, 9, 27]
    layer_size_list = [512, 2048, 8192]
    batch_size_list = [64, 256, 1024]
    learning_rate_list = [1e-5, 1e-4, 1e-3]
    # 2. Base directory for all experiments:
    #    We'll append "MLP_experiments" to the existing config_template.save_dir.
    base_save_dir = os.path.join(
        config_template.save_dir,
        "unconditioned_MLP_experiments" if unconditioned else "MLP_experiments",
    )
    os.makedirs(base_save_dir, exist_ok=True)
    if not merge_only:

        # Build the Cartesian product of hyperparam combos
        all_combos = list(
            product(
                num_layers_list, layer_size_list, batch_size_list, learning_rate_list
            )
        )

        # ---------------------------
        # 2. LOAD TOKENIZER & PROCESSOR
        # ---------------------------
        # logger.info(f"Loading tokenizer for base model: {config.base_model_name}")
        # tokenizer = AutoTokenizer.from_pretrained(config_template.base_model_name, trust_remote_code=True)
        # data_processor = HierarchicalDataProcessor(
        #     tokenizer=tokenizer,
        #     tokenizer_kwargs=config_template.tokenizer_kwargs,
        #     sequence_column=config_template.sequence_column,
        #     taxonomic_ranks=config_template.taxonomic_ranks,
        #     save_file=config_template.data_processor_filename,
        # )
        # data_processor.load_processor(config_template.data_processor_path)

        # ---------------------------
        # 3. LOAD EMBEDDINGS (TRAIN, VAL, TEST)
        # ---------------------------

        train_embeddings, train_labels = read_group_in_batches(
            config_template.embeddings_checkpoint_h5, "train"
        )
        val_embeddings, val_labels = read_group_in_batches(
            config_template.embeddings_checkpoint_h5, "val"
        )
        test_embeddings, test_labels = read_group_in_batches(
            config_template.embeddings_checkpoint_h5, "test"
        )

        # ranks = data_processor.taxonomic_ranks  # e.g. ["superkingdom", "phylum", "genus"]
        # logger.info(f"Taxonomic ranks: {ranks}")

        train_labels = np.asarray(train_labels)
        val_labels = np.asarray(val_labels)
        test_labels = np.asarray(test_labels)

        # ---------------------------
        # 4. BUILD DATASETS & LOADERS
        # ---------------------------
        logger.info(f"Batch size: {config_template.train_batch_size}")
        train_ds = HierarchicalDataset(train_embeddings, train_labels)
        val_ds = HierarchicalDataset(val_embeddings, val_labels)
        test_ds = HierarchicalDataset(test_embeddings, test_labels)

        train_loader = DataLoader(
            train_ds, batch_size=config_template.train_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=config_template.train_batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=config_template.train_batch_size, shuffle=False
        )

        data_loaders = [train_loader, val_loader, test_loader]

        # We'll store data about each experiment in `results_list`
        for nl, ls, bs, lr in all_combos:
            exp_name = f"exp_numLayers{nl}_layerSize{ls}_bs{bs}_lr{lr}"
            exp_folder = os.path.join(base_save_dir, exp_name)

            # 1. Check if experiment is finished
            finished_marker = os.path.join(exp_folder, "finished.txt")
            summary_csv = os.path.join(exp_folder, "test_metrics_summary.csv")
            if os.path.isfile(finished_marker) or os.path.isfile(summary_csv):
                print(f"[INFO] Skipping {exp_name}: already completed.")
                continue

            # Otherwise, create folder if it doesn't exist
            os.makedirs(exp_folder, exist_ok=True)

            # 2. Make a copy of the config so we don't overwrite
            config = copy.deepcopy(config_template)

            # 3. Assign the hyperparams
            config.num_layers = nl
            config.layer_size = ls
            config.train_batch_size = bs
            config.learning_rate = lr

            # 4. Re-point save_dir and test_metrics_dir to this experiment folder
            config.save_dir = exp_folder
            config.test_metrics_dir = exp_folder

            # 5. Run the training/evaluation
            print(f"=== Running experiment: {exp_name} ===")
            print(f"num_layers={nl}, layer_size={ls}, batch_size={bs}, lr={lr}")

            main(config, data_loaders, unconditioned)
            print(f"=== Finished experiment: {exp_name} ===")

            # 6. Mark experiment as finished
            with open(finished_marker, "w") as f:
                f.write("Done!\n")

    # 7. After running all experiments, gather results
    merged_csv = os.path.join(
        base_save_dir,
        f"{os.path.basename(config_template.save_dir)}_all_experiments_merged.csv",
    )
    results_list = gather_all_results(
        base_save_dir,
        num_layers_list,
        layer_size_list,
        batch_size_list,
        learning_rate_list,
    )

    if results_list:
        df_merged = pd.DataFrame(results_list)
        df_merged.to_csv(merged_csv, index=False)
        print(f"\n[INFO] Merged results saved to {merged_csv}")
    else:
        print("\n[INFO] No results found to merge.")


def gather_all_results(
    base_dir, num_layers_list, layer_size_list, batch_size_list, learning_rate_list
):
    """
    For each hyperparam combo, if test_metrics_summary.csv exists,
    parse it and extract relevant info for each taxonomic level (row).
    Returns a list of dicts, one per tax level per experiment.
    """
    from itertools import product
    import pandas as pd
    import os

    results = []

    for nl, ls, bs, lr in product(
        num_layers_list, layer_size_list, batch_size_list, learning_rate_list
    ):
        exp_name = f"exp_numLayers{nl}_layerSize{ls}_bs{bs}_lr{lr}"
        exp_folder = os.path.join(base_dir, exp_name)

        summary_csv = None
        for root, dirs, files in os.walk(exp_folder):
            if "test_metrics_summary.csv" in files:
                summary_csv = os.path.join(root, "test_metrics_summary.csv")
                break

        if summary_csv is None:
            print(f"[DEBUG] No test_metrics_summary.csv found under: {exp_folder}")
            continue

        try:
            df = pd.read_csv(summary_csv, index_col=0)

            if df.empty:
                print(f"[DEBUG] Empty CSV: {summary_csv}")
                continue

            # Clean index in case of whitespace
            df.index = df.index.str.strip()

            for tax_level in df.index:
                row = df.loc[tax_level]

                # Ensure the row is a Series, not a DataFrame slice
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                row_data = {
                    "num_layers": nl,
                    "layer_size": ls,
                    "batch_size": bs,
                    "learning_rate": lr,
                    "tax_level": tax_level,
                }

                # Some rows like "Overall" may not have all columns, use .to_dict()
                row_data.update(row.to_dict())
                results.append(row_data)

        except Exception as e:
            print(f"[WARN] Failed to parse {summary_csv}: {e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Driver for multiple MLP experiments")
    parser.add_argument(
        "config_file",
        type=str,
        default="bertax/sample/train_embeddings.yaml",
        nargs="?",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--unconditioned", action="store_true", help="Use the unconditioned MLP."
    )
    parser.add_argument(
        "--merge_only", action="store_true", help="Only merge the results."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load base config from command line path
    config_file = args.config_file
    config_path = os.path.join(CONFIG_DIR, config_file)
    print(f"[INFO] Loading config from {args.config_file}")
    config_template = Config(config_path)

    # Possibly override other fields here if needed
    # e.g. config_template.num_epochs = 20

    run_all_experiments(config_template, args.unconditioned, args.merge_only)
