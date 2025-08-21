**COPYRIGHT NOTICE**

© 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

**MicrobeRT**: Leveraging Language Models for Analysis of Metagenomic Sequencing Data

This repository contains a comprehensive and configurable pipeline for fine-tuning pre-trained genomic language models (gLMs) on labels of interest such as taxonomic hierarchy and evaluating models and tracking experimental results. See [acknowledgement](#acknowledgement).

# Instructions

There are two primary training/testing scripts for fine-tuning and evaluating open-source genomic language models. 

- **Fine-tuning** (`train_model.py`): Fine-tunes a genomic language model on either a hierarchical multiclass classification (e.g. taxonomy prediction) or standard multiclass classification task of your choice, and evaluates the fine-tuned model on a test set of specification.  
- **Generating Embeddings** (`train_embeddings.py`): Generates embeddings from a genomic language model. 

## Setup Instructions

**Requirements:** 
- Python 3.11
- Open source Python libraries defined in `requirements.txt`
> Note: All software packages used in this project fall under the MIT License, Apache License, or BSD License. and are therefore susceptible to release under these open-source licenses. See `requirement_licenses.md` for more details on requirements and their licenses. All Genomic Language Models considered are open source and available on public model repostory Hugging Face. See `requirement_models.md` for list of models used. 

### Create a Virtual Environment or Conda Environment

```
# Starting from the Project's ROOT DIRECTORY
python -m venv env_name
```
or 
```
conda create --prefix ENV_NAME python=3.11
```

Activate virtual environment:

- **Linux/macOS:** `source env_name/bin/activate`
- **Windows:** `env_name\Scripts\activate`

or 
```
conda activate analysis
```

### Install Dependencies
```
pip install -e . --default-timeout=1000
```
This installs `analysis` as a module that enables local imports, e.g.
```
from analysis.experiment.models.hierarchical_model import HierarchicalClassificationModel
```

## Scripts

All of the following scripts uses a single parameter, a path to a config `yaml` file as an argument. See ### Config section for details. 

### Fine-tuning
To fine-tune a genomic language model and evaluate it on a test set, run 
```
python ~/analysis/experiment/train_model.py --config_path CONFIG_YAML_RELATIVE_PATH
```
### Embedding
To only generate model embeddings, run: 
```
python ~/analysis/experiment/train_embeddings.py --config_path CONFIG_YAML_RELATIVE_PATH
```

<!-- ### Baseline
To run a baseline evaluating pre-trained models with hierarchical multiclass classification or standard multiclass classification head layers without fine-tuning, run 
```
python ~/analysis/experiment/eval_pretrained.py --config_path CONFIG_YAML_RELATIVE_PATH
``` -->

<!-- ### Transfer Learning
Additional scripts for training a downstream classification model (e.g. Logistic Regression or a Random Forest) trained on model generated embeddings as features are contained in `~/analysis/experiment/transfer_learning`. This transfer learning pipeline is distinct for hierarchical classification and our implementation uses the package `hiclass`. These can be run as follows:
```
python ~/analysis/experiment/transfer_learning/train_hiclass --config_path CONFIG_YAML_RELATIVE_PATH
```
for hierarchical multiclass classification.
```
python ~/analysis/experiment/transfer_learning/train_class --config_path CONFIG_YAML_RELATIVE_PATH
```
for multiclass classification.  -->

## Config

The `Config` class `~/analysis/experiment/utils/config` reads in the path to a `yaml` containing all necessary training, model, and file path parameters. These values override the default settings defined in the class.

Example usage:

```python
from analysis.experiment.utils.config import Config
config = Config("path/to/your_config.yaml")
```

Your `yaml` file should define any parameters you want to customize. You don’t need to include every key—any omitted values will fall back to class defaults.

### Example `config.yaml`

```yaml
training_data: "data/train.csv"
validation_data: "data/val.csv"
testing_data: "data/test.csv"
stratify: "genus"

classification: true
label_column: "genus"
sequence_column: "sequence"

model_type: "NT"
base_model_name: "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer_name: "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

train_batch_size: 32
eval_batch_size: 32
epochs: 5
learning_rate: 0.0001
fp16: true

experiment_name: "GenusClassifier"
```

## Configuration Parameters

Here is a breakdown of the most salient config parameters that can be specified. 
For a full detailed list, see documentation in ~/analysis/experiment/utils/config.py`

### Data Paths
| Parameter | Description |
|----------|-------------|
| `training_data`, `testing_data` | Paths to raw CSV files |
| `stratify` | Optional column name to stratify train/val/test splits |
| `tokenized_training_data`, `tokenized_testing_data` | Paths to preprocessed/tokenized dataset (defaults to location saved by default) |
| `data_processor_path` | Path to saved data processor `.pkl` for tokenizing and encoding input train/test data (defaults to `save_dir`) |

### Task & Processing
| Parameter | Description |
|----------|-------------|
| `classification` | `true` for multiclass classification, `false` for hierarchical classification |
| `label_column` | Column name of input data csv containing label for classification |
| `sequence_column` | Column name of input data csv contaning DNA sequence as a string, defaults to `sequence`  |

### Model Setup
| Parameter | Description |
|----------|-------------|
| `model_type` | Model category, must be either `NT`, `DNABERT`, `HYENA`, `METAGENE`, or `GenomeOcean` depending on `base_model_name` |
| `base_model_name` | HuggingFace model name, defaults to `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`  |
| `tokenizer_name` | Tokenizer model name, defaults to `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` |
| `tokenizer_kwargs` | Optional dict of tokenizer arguments |

### PEFT
Use these parameters if using Parameter-Efficient Fine-Tuning (PEFT) methods available to fine-tune models.
| Parameter | Description |
|----------|-------------|
| `peft_method` | `"lora"`, `"ia3"`, or `None` |
| `lora_r` | Rank for LoRA if enabled |

### Training Parameters
| Parameter | Description |
|----------|-------------|
| `train_batch_size`, `eval_batch_size` | Per-GPU batch sizes |
| `epochs` | Total number of epochs |
| `learning_rate` | Initial learning rate |
| `fp16`, `bf16` | Mixed precision options |
| `weight_decay`, `warmup_ratio` | Optimization params |
| `eval_accumulation_steps` | Batches to accumulate before eval |
| `multi_gpu_count` | Number of GPUs used |
| `early_stopping_patience` | Epochs to wait with no improvement |

### Evaluation
| Parameter | Description |
|----------|-------------|
| `metric_for_best_model` | Metric to optimize (e.g., `"eval_loss"`) |
| `greater_is_better` | If `true`, higher metric = better model |

### Saving & Logging
| Parameter | Description |
|----------|-------------|
| `experiment_name` | Name of directory for saving all resulting outputs such as model checkpoints, test results, etc.  for this run (auto-generated if none specifieid) |
| `testing_name` | Subfolder name for saving test results, defaults to `test_results` |

---

## Configuration Tips

- You **must** define either a training dataset path or a testing dataset path in the config `yaml` to ensure that training or evaluation mode is run. 
- Make sure `experiment_dir` is writable (currently hardcoded to `~/analysis/experiment` in the class).
- If not explicitly specified, `experiment_name` defaults to a random set of 24 characters as a unique identifier of the experiment run.
- By default, all experiment runs are output to the directory `~/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/`.
- All logs from training runs are saved to a single, timestamped log file in the experiment's save directory `~/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/`.
- `testing_name` is name of subdirectory that stores results of a given test run, defaults to `test_results`. 
- Outputs of the experiment run include label predictions, all label probabilities, experimental run arguments, saved off data_processor `.pkl` file. All model checkpoints, model training history values and metric plot, and model performance on test dataset are stored in `~/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/` and subfolders within.
- The best model is saved off as `~/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/model.safetensors`.
- Model checkpoints and training history are stored in `~/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/train_history`.
- Model test results are stored in `~/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/test_results`
- If path to training data is specified in `training_data`, then the script runs in `training` mode and trains on a subset of this data. If `validation_data` and/or `testing_data` are also specified, then best trained model will evaluate on the datasets specified by these path. Otherwise, the validation and/or testing dataset will be sampled from the training_data (80%/10%/10% split). Option for stratifying the train/val/test split is contained in the condfig parameter `stratify`, which must be a valid input data column. 
- If `training_data` is not specified, then the script is not in `training` mode and does train on any dataset. Instead, the top performing model from the specified `experiment_name` run is loaded in used to evaluate on the testing_data specified in `testing_data`. Note that `testing_data` must be specified in this case to ensure that eval is done on a valid test set.
- `tokenizer_kwargs` is a dict of optional keyword arguments passed into the tokenizer.
- List of possible `model_types`, `tokenizer_name`, and `base_model_name` are contained in `analysis.experiment.utils.constants.MODELS`.
- If doing a hierarchical classification, make sure config specifies `classification` as `False`. Also make sure`taxonomic_ranks` is specified from list of possible taxonomic ranks are contained in  `analysis.experiment.utils.constants.TAX_RANKS`
- If doing a multiclass classification, make sure config specifies `classification` as `True`. Also make sure`label_column` is specified as column to be classified.
- Label encoding and dataset tokenization are cached, and tokenized datasets are saved for faster reloads when rerunning the same training script. If you specify `tokenized_training_data` , `tokenized_validation_data`  and/or `tokenized_testing_data` parameters in the config, then it will use the datasets corresponding to those filepaths (directories) explicitly. Otherwise, it will check in the default location for these datasets if they exist, and load those in. If those also dont exist, it tokenizes from scratch and saves off in the default location.
- If a model was trained in a previous run, subsequent runs will automatically resume from the last saved checkpoint—preemptibility is enabled to support seamless continuation of training.

## Generated Directory Structure

Once a config file is loaded and a train run is initiated, the following structure is created:

```
experiment/
└── runs/
    └── <EXPERIMENT_NAME>/
        └── <MODE_NAME>/
            ├── train_history/
            ├── models/
            ├── test_results/
            │   └── tokenized_testing_data/
            ├── tokenized_training_data/
            ├── tokenized_validation_data/
            └── config_arguments.txt
            └── data_processor.pkl
            └── data_processor.pkl
            └── train_DATE_TIME_.log


```


## Obtaining Class Metrics
Evaluation metrics per individual class can be generated by running the script
`~/analysis/experiment/utils/metrics_generator.py`

Example usage: 
```
parent_dir = "/home/apluser/analysis/analysis/experiment/RUN_DIR"
generator = MetricsGenerator(None)
generator.process_multiple_models(parent_dir)
```

This will generate a `class_metrics.csv` file in each model output directory, containing the F1 score, precision, recall, and support for each class with at least one example in the test set.

Additional features of the class:
- Identifying the correlation between F1 score and support.
- Generating a confusion matrix.
- Calculating the Jaccard similarity between the predictions of each model in the parent_dir.

### Triton Issue
It looks like DNABERT-2 is not compatible with the `triton` package [DNABERT ISSUE](https://github.com/MAGICS-LAB/DNABERT_2/issues/57).
We got around this by explicitly uninstalling `pip uninstall triton`.

### Acknowledgement

This work was supported by funding from the U.S. Centers for Disease Control and Prevention (CDC) through the Office of Readiness and Response under Contract No. 75D30124C20202.
