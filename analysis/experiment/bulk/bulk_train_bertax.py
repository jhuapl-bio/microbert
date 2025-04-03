# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


# !/usr/bin/env python3

import os
import sys
import re
import yaml
import json
import subprocess
import torch

# from analysis.experiment.utils.constants import MODELS

# removing NT since already down
models = {
    "NT": [
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    ],
    "DNABERT": [
        "zhihan1996/DNABERT-2-117M",
        "zhihan1996/DNABERT-S",
    ],
    "HYENA": [
        "LongSafari/hyenadna-large-1m-seqlen-hf",
        "LongSafari/hyenadna-medium-450k-seqlen-hf",
        "LongSafari/hyenadna-medium-160k-seqlen-hf",
        "LongSafari/hyenadna-small-32k-seqlen-hf",
    ],
}

# File path for the JSON file containing maximum batch sizes
max_batch_sizes_file_path = (
    "/home/apluser/analysis/analysis/experiment/runs/batch_size/max_batch_sizes.json"
)

# Load the JSON file into a Python dictionary
with open(max_batch_sizes_file_path, "r") as json_file:
    max_batch_sizes = json.load(json_file)

# File path for the JSON file containing learning rates
learning_rates_file_path = (
    "/home/apluser/analysis/analysis/experiment/runs/batch_size/learning_rates.json"
)

# Load the JSON file into a Python dictionary
with open(learning_rates_file_path, "r") as json_file:
    learning_rates = json.load(json_file)

# File path for the JSON file containing maximum evaluation batch sizes
max_eval_batch_sizes_file_path = "/home/apluser/analysis/analysis/experiment/runs/batch_size/max_eval_batch_sizes.json"

# Load the JSON file into a Python dictionary
with open(max_eval_batch_sizes_file_path, "r") as json_file:
    max_eval_batch_sizes = json.load(json_file)

experiment_name = "bertax/full"

for model_type in models:
    model_names = models[model_type]
    for model_name in model_names:
        name = model_name[model_name.index("/") + 1 :]
        file_path = f"/home/apluser/analysis/analysis/experiment/configs/{experiment_name}/{name}.yaml"
        data = {
            "training_data": "/home/apluser/analysis/analysis/process_bertax/bertax_out_split/train.tsv",
            "testing_data": "/home/apluser/analysis/analysis/process_bertax/bertax_out_split/test.tsv",
            # Model Parameters
            "taxonomic_ranks": ["superkingdom", "phylum", "genus"],
            "model_type": model_type,
            "tokenizer_name": model_name,
            "base_model_name": model_name,
            # Training Parameters
            "train_batch_size": max_batch_sizes[model_name],
            "eval_batch_size": max_eval_batch_sizes[model_name],
            "learning_rate": learning_rates[model_name],
            "epochs": 5,
            # Save Directory
            "experiment_name": experiment_name,
        }

        with open(file_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)

for model_type in models:
    model_names = models[model_type]
    for model_name in model_names:
        # Clear GPU memory
        torch.cuda.empty_cache()

        name = model_name[model_name.index("/") + 1 :]
        config_file = f"{experiment_name}/{name}.yaml"

        # Path to the virtual environment activation script
        conda_env = "analysis"

        # Path to the Python script
        script_path = "../train_model.py"

        # Command to source .bashrc, activate Conda environment, and run the script
        command = f"source ~/.bashrc && conda activate {conda_env} && python {script_path} {config_file}"
        subprocess.run(
            command,
            shell=True,  # Allow shell commands like `source`
            executable="/bin/bash",  # Use bash to execute commands
            check=True,  # Raise exception on failure
        )
