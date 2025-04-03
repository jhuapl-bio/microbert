# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


# !/usr/bin/env python3

import os
import sys
import re
import yaml
import json
import subprocess
import torch

models = {
    "NT": [
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    ],
    # "DNABERT": [
    #     "zhihan1996/DNABERT-2-117M",
    #     "zhihan1996/DNABERT-S",
    # ],
    # don't need Hyena models for paper - due to code update, previously trained models will not predict with new forward method)
    # "HYENA": [
    #     "LongSafari/hyenadna-large-1m-seqlen-hf",
    #     "LongSafari/hyenadna-medium-450k-seqlen-hf",
    #     "LongSafari/hyenadna-medium-160k-seqlen-hf",
    #     "LongSafari/hyenadna-small-32k-seqlen-hf",
    # ],
}


# File path for the JSON file containing maximum evaluation batch sizes
max_eval_batch_sizes_file_path = "/home/apluser/analysis/analysis/experiment/runs/batch_size/results/all/max_eval_batch_sizes.json"

# Load the JSON file into a Python dictionary
with open(max_eval_batch_sizes_file_path, "r") as json_file:
    max_eval_batch_sizes = json.load(json_file)

# TODO modify to include all experiments ever run thus far
# peft4.2 and peft4.3 do not have data processor pkl files in them
# none of the bertax substratified have data processor pkl files in them
# "bertax_substratified/full", "bertax_substratified/peft_lora", "bertax_substratified/peft_ia3", "bertax_substratified/partial_frozen", "bertax_substratified/full", 
experiment_names = ["peft_lora4_rerun", "peft_lora4.2_rerun", "peft_lora4.3_rerun"]

os.makedirs(os.path.join("/home/apluser/analysis/analysis/experiment/runs/bertax", "no_unknowns"), exist_ok=True)
os.makedirs(os.path.join("/home/apluser/analysis/analysis/experiment/configs/bertax", "no_unknowns"), exist_ok=True)

for experiment_name in experiment_names:
    existing_run_dir = os.path.join("/home/apluser/analysis/analysis/experiment/runs/bertax", experiment_name)
    run_model_names = [
        name for name in os.listdir(existing_run_dir)
        if os.path.isdir(os.path.join(existing_run_dir, name)) and not name.startswith(".")
    ]
    for model_type in models:
        model_names = models[model_type]
        for model_name_full in model_names:
            model_name = model_name_full[model_name_full.index('/') + 1:]
            if model_name not in run_model_names:
                continue
            test_metrics_dir = os.path.join("/home/apluser/analysis/analysis/experiment/runs/bertax", 
                                            "no_unknowns", 
                                            experiment_name, 
                                            model_name)
            config_dir = f"/home/apluser/analysis/analysis/experiment/configs/bertax/no_unknowns/{experiment_name}"
            os.makedirs(config_dir, exist_ok=True)
            file_path = os.path.join(config_dir, f"{model_name}.yaml")
            data = {
                "testing_data": "/home/apluser/analysis/analysis/process_bertax/bertax_out_split/test_nounknown.csv", # testing on unknown
                "testing_name": "no_unknown", # testing on unknown set
                "new_test_run": True,
                "test_metrics_dir": test_metrics_dir,
                # Model Parameters
                "taxonomic_ranks": ["superkingdom", "phylum", "genus"],
                "model_type": model_type,
                "tokenizer_name": model_name_full,
                "base_model_name": model_name_full,
                "eval_batch_size": max_eval_batch_sizes[model_name_full],
                # Save Directory - should be old save directory
                "experiment_name": f"bertax/{experiment_name}",
            }
    
            with open(file_path, "w") as file:
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)
                print(f'created config file: {file_path}')


for experiment_name in experiment_names:
    existing_run_dir = os.path.join("/home/apluser/analysis/analysis/experiment/runs/bertax", experiment_name)
    
    run_model_names = [
        name for name in os.listdir(existing_run_dir)
        if os.path.isdir(os.path.join(existing_run_dir, name)) and not name.startswith(".")
    ]
    for model_type in models:
        model_names = models[model_type]
        for model_name_full in model_names:
            model_name = model_name_full[model_name_full.index('/') + 1:]
            if model_name not in run_model_names:
                continue
            # Clear GPU memory
            torch.cuda.empty_cache()

            config_dir = f"/home/apluser/analysis/analysis/experiment/configs/bertax/no_unknowns/{experiment_name}"
            config_file = os.path.join(config_dir, f"{model_name}.yaml")    
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
