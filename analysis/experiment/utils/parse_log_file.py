import os
import re
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def parse_log_file(filename, test_metrics_name="test_results"):
    cwd = os.getcwd()
    cwd_filename = os.path.normpath(os.path.join(cwd, filename))
    #
    split_cwd = os.path.split(
        os.path.split(re.sub(f"/{test_metrics_name}", "", cwd_filename))[0]
    )
    exp_name = os.path.split(split_cwd[0])[1]
    model_name = split_cwd[1]
    if re.search(".log", filename):
        f = open(filename, "r")
        data = f.read()
        data = data.split("\n")
        times = []
        start_results = False
        metrics_results = {}
        data_row = None
        for row in data:
            if re.search("Number of trainable parameters", row):
                trainable_parameters = int(re.sub(",", "", row.split(" = ")[-1]))
            if re.search("Epoch [0-9]* completed in", row):
                times.append(float(row.split("completed in ")[1].split(" ")[0]))
            if start_results:
                level_split = "Summary Metrics on test set for column"
                if re.search(level_split, row):
                    if data_row is not None:
                        metrics_results[field] = data_row
                    data_row = {}
                    field = row.split(level_split + " ")[-1][0:-1]
                elif len(re.findall(": ", row)) > 1:
                    metrics = row.split("INFO :  ")[-1].split(": ")
                    data_row[metrics[0]] = float(metrics[1])
                else:
                    pass
            if re.search("\*\*\*\*\* Running Prediction \*\*\*\*\*", row):
                start_results = True
        #
        if len(data_row) > 0:
            metrics_results[field] = data_row
        metrics_results = pd.DataFrame(metrics_results).transpose()
        config_file = os.path.join(
            os.path.split(cwd_filename)[0], "config_arguments.txt"
        )
    elif re.search(".csv", filename):
        metrics_results = pd.read_csv(filename)
        times = np.nan
        trainable_parameters = np.nan
        metrics_results.index = metrics_results["Unnamed: 0"]
        metrics_results = metrics_results.drop(columns="Unnamed: 0")
        config_file = os.path.join(
            os.path.split(os.path.split(cwd_filename)[0])[0], "config_arguments.txt"
        )
    g = open(config_file, "r")
    config = g.read()
    conditions = {}
    if re.search("\[[ \n\S]*\]", config):
        config_fix = ""
        start = 0
        while len(config) > 0:
            x = re.search("\[[ \n\S]*\]", config)
            if x:
                config_fix += config[start : x.span()[0]]
                config_fix += str(config[x.span()[0] : x.span()[1]].split('"')[1::2])
                config = config[x.span()[1] :]
            else:
                config_fix += config
                config = ""
        config = config_fix
    config = re.sub('{|}|"', "", config).split("\n")[1:-1]
    for cond in config:
        cond = re.sub(" *|,$", "", cond)
        split_cond = cond.split(":")
        conditions[split_cond[0]] = split_cond[1]
    #
    results = []
    for r, df in metrics_results.iterrows():
        df.index = df.name + df.index
        df = pd.DataFrame(df).transpose()
        df.index = [0]
        results.append(df)
    metrics_results = pd.concat(results, axis=1)
    metrics_results["average epoch time"] = [np.mean(times)] * len(metrics_results)
    metrics_results["file name"] = [cwd_filename] * len(metrics_results)
    metrics_results["trainable_parameters"] = [trainable_parameters] * len(
        metrics_results
    )
    metrics_results["experiment name"] = [exp_name] * len(metrics_results)
    metrics_results["model name"] = [model_name] * len(metrics_results)
    keep_conditions = [
        "training_data",
        "testing_data",
        "randomization",
        "freeze_layers_fraction",
        "epochs",
        "learning_rate",
        "fp16",
        "weight_decay",
        "warmup_ratio",
    ]
    for k in keep_conditions:
        if k in conditions:
            metrics_results[k] = [conditions[k]] * len(metrics_results)
    return metrics_results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--test_metrics_name", type=str, default="test_results")
    parser.add_argument("--metrics_file", type=str)

    args = parser.parse_args()
    search_name = args.test_metrics_name

    # If args.log_file is a single file path
    if not os.path.isdir(args.log_file):
        args.log_file = [args.log_file]
    else:
        # Assume it's a directory, and search for test_results inside it
        args.log_file = [
            os.path.join(root, "test_metrics_summary.csv")
            for root, _, _ in os.walk(args.log_file)
            if re.search(rf"{search_name}$", root)
        ]
log_files = args.log_file
log_files = [i for i in args.log_file if any(sub in i for sub in ['dnabert_lr'])]

if os.path.exists(args.metrics_file):
    os.remove(args.metrics_file)

for log_file in log_files:
    try:
        metrics = parse_log_file(log_file)
        if os.path.exists(args.metrics_file):
            metrics.to_csv(args.metrics_file, mode="a", header=False)
        else:
            metrics.to_csv(args.metrics_file)
    except Exception as e:
        print(f"Skipping {log_file} due to error: {e}")
