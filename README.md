COPYRIGHT NOTICE

© 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

MicrobeRT: Leveraging Language Models for Analysis of Metagenomic Sequencing Data

This repository contains a comprehensive and configurable pipeline for fine-tuning pre-trained genomic language models (gLMs) on labels of interest such as taxonomic hierarchy and evaluating models and tracking experimental results. This work was supported by funding from the U.S. Centers for Disease Control and Prevention through the Office of Readiness and Response under Contract # 75D30124C20202.

# Setup and run from source code

**Requirements:** 
- Python>=3.11
- Python libraries defined in `requirements.txt`
> Note: All software packages used in this project fall under the MIT License, Apache License, or BSD License. and are therefore susceptible to release under these open-source licenses. See `requirement_licenses.md` for more details on requirements and their licenses. All Genomic Language Models considered are open source and available on public model repostory Hugging Face. See `requirement_models.md` for list of models used. 

### Environment
Create a virtual environment:
```
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

#### Triton / FlashAttention Compatibility
DNABERT-2 currently has compatibility issues with the triton package when running on certain hardware, such as the NVIDIA H100: [related issue](https://github.com/MAGICS-LAB/DNABERT_2/issues/57)
To work around this, we explicitly uninstalled Triton using: `pip uninstall triton`
This issue did not occur when running on an NVIDIA A100.

### Model Compatibility
We have verified that the training and testing pipeline functions correctly with the following models, and the pipeline will raise a 

#### Nucleotide Transformer (NT)
- [InstaDeepAI/nucleotide-transformer-v2-50m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)  
- [InstaDeepAI/nucleotide-transformer-v2-100m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-100m-multi-species)  
- [InstaDeepAI/nucleotide-transformer-v2-250m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-250m-multi-species)

#### DNABERT
- [zhihan1996/DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M)  
- [zhihan1996/DNABERT-S](https://huggingface.co/zhihan1996/DNABERT-S)

#### HyenaDNA
- [LongSafari/hyenadna-large-1m-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen-hf)  
- [LongSafari/hyenadna-medium-450k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-medium-450k-seqlen-hf)  
- [LongSafari/hyenadna-medium-160k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf)  
- [LongSafari/hyenadna-small-32k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen-hf)

#### METAGENE
- [metagene-ai/METAGENE-1](https://huggingface.co/metagene-ai/METAGENE-1)

#### GenomeOcean
- [pGenomeOcean/GenomeOcean-4B](https://huggingface.co/pGenomeOcean/GenomeOcean-4B)  
- [pGenomeOcean/GenomeOcean-100M](https://huggingface.co/pGenomeOcean/GenomeOcean-100M)  
- [pGenomeOcean/GenomeOcean-500M](https://huggingface.co/pGenomeOcean/GenomeOcean-500M)


## Scripts

There are four primary training/testing scripts for fine-tuning and evaluating open-source genomic language models. 
- **Generate Data** (`train_data.py`): Generates tokenized data for a given genomic language model
- **Fine-tuning** (`train_model_multi_gpu.py`): Fine-tunes a genomic language model on a multiclass classification task of your choice (e.g. taxonomic classification), and evaluates the fine-tuned model on a test set of specification.  
- **Generating Embeddings** (`train_embeddings.py`): Generates embeddings from a genomic language model. 
- **Inference** (`test_sequences.py`): Generates predictions of a trained gLM on a set of test sequences. 

Most scripts use a single parameter: the path to a config `yaml` file as an argument.  
See the [Config](#config) and [Config Parameters](#config-parameters) sections for details.

### Data Tokenization
To generate tokenized data only for a set of sequences
```
python ~/analysis/analysis/experiment/train_data.py --config_path CONFIG_YAML_RELATIVE_PATH
```

### Model Fine-tuning
To fine-tune a genomic language model on a train set of sequences and evaluate it on a test set of sequences, run 
```
python ~/analysis/analysis/experiment/train_model_multi_gpu.py --config_path CONFIG_YAML_RELATIVE_PATH
```

### Generating Embeddings
To generate model embeddings for a set of sequences, run: 
```
python ~/analysis/analysis/experiment/train_embeddings.py --config_path CONFIG_YAML_RELATIVE_PATH
```

### Batched Inference

We provide a script to evaluate classifications on input FASTA/FASTQ sequences using a trained genomic language model.  
The script loads a saved `DataProcessor`, model weights, and the base model tokenizer. 
It processes sequences in batches and outputs predictions per label, returning and saving the results to an output JSON file.
```
python /analysis/experiment/test_sequences.py \
    --input-path <INPUT_FASTA_FILE> \
    --output-path <OUTPUT_JSON_FILE> \
    --model-dir <MODEL_DIR> \
    [--use-gpu] \
    --batch-size <BATCH_SIZE> \
    --top-k <TOP_K> \
    --threshold <THRESHOLD>
```
#### Model download
In order to use `test_sequences.py` you must have access to a `data_processor.pkl` file, the base model from Hugging Face, and trained model weights file.
This must be organized with the following structure:
```
MODEL_DIR/
  └── base_model/
  └── data_processor/
  └── model/
```
These files and models can be downloaded from [HuggingFace](https://huggingface.co/jhuapl-bio).
This should be placed in an appropriate local directory whose path is referenced with the argument `--model-dir`.
- **`MODEL_DIR/base_model`**  
  Holds the base_model and tokenizer files necessary for preprocessing input sequences and loading fine-tuned model.  
- **`MODEL_DIR/data_processor`**
  Contains the data processor used to store and encode model inference labels.
- **`MODEL_DIR/model`**  
  Contains the trained model weight files (e.g., `model.safetensors`) that are loaded in for evaluating sequences.
- 
#### Inference Arguments

| Argument        | Description                                                                          |
|-----------------|--------------------------------------------------------------------------------------|
| `--input-path`  | Path to the input FASTA/FASTQ file (`.fa` / `.fq`, optionally compressed with `.gz`) |
| `--output-path` | Path where predictions will be saved as a JSON file                                  |
| `--model-dir`   | Directory where data processor, base model tokenizer, and trained model lives        |
| `--use-gpu`     | Optional flag to enable GPU inference if available                                   |
| `--batch-size`  | Number of sequences per batch (default: `256`)                                       |
| `--top-k`       | Number of top predictions per label to return (default: `5`)                         |
| `--threshold`   | Minimum probability required to include a prediction (default: `0.2`)                |

#### Transfer Learning
Additional scripts for training a downstream classification model (e.g. Random Forest or MLP) trained on model generated embeddings as features are contained in `~/analysis/analysis/experiment/transfer_learning`. 
Note this transfer learning pipeline for hierarchical classification uses the package `hiclass`. 

### Data Processor

Before data can be tokenized and used for training with these scripts, it must first be label-encoded using a  
`DataProcessor` object from `analysis.experiment.utils.data_processor.DataProcessor`.
This preprocessing step is best performed in a separate script, which prepares the raw dataset and fits the label encoder.
Example preprocessing scripts can be found in:
- `~/analysis/analysis/process_taxonomy`
- `~/analysis/analysis/process_amr`

### Example

```python
from analysis.experiment.utils.data_processor import DataProcessor

# Initialize DataProcessor
data_processor = DataProcessor(
    sequence_column=SEQUENCE_COL,   # str: name of column in dataframe containing sequences
    labels=LABEL_COLS,              # list[str]: list of label/taxonomy columns
    save_file="data_processor.pkl", # str: filename to save the fitted processor
)

# Fit label encoder on a dataframe and save results
df = data_processor.fit_encoder_from_df(
    df,                 # pandas.DataFrame: your preprocessed dataset
    save_directory=DATA_DIR  # str or Path: directory where processor artifacts are saved
)
```

## Config

The `Config` class `~/analysis/analysis/experiment/utils/config` reads in the path to a `yaml` containing all necessary training, model, and file path parameters. These values override the default settings defined in the class.

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
For a full detailed list, see documentation of parameters listed in `~/analysis/analysis/experiment/utils/config.py`

### Data Paths
| Parameter | Description |
|----------|-------------|
| `training_data`, `validation_data`, `testing_data` | Paths to raw CSV files |
| `stratify` | column name, if any, to stratify train/val/test splits |
| `testing_name` | Subfolder name for saving test results, defaults to `test_results` |
| `test_metrics_dir` | Subfolder name for saving test metrics if different from original train run | 
| `new_test_run` | If testing on new dataset different from original train run, should be set to `True`  | 
| `tokenized_training_data`, `tokenized_validation_data`, `tokenized_testing_data` | Paths to preprocessed/tokenized dataset (defaults to location saved by default) |
| `data_processor_path` | Path to saved data processor `.pkl` for encoding input train/test data (defaults to `save_dir`) |
| `data_processor_filename` | Name of data processor `.pkl` file, defaults to `data_processor.pkl` |

### Task & Processing
| Parameter | Description |
|----------|-------------|
| `sequence_column` | Column name of input data csv contaning DNA sequence as a string, defaults to `sequence`  |
| `labels` | Column name sof input data csv containing label for classification, defaults to `['superkingdom', 'phylum', 'genus']` |

### Model Setup
| Parameter | Description |
|----------|-------------|
| `model_type` | Model category, must be either `NT`, `DNABERT`, `HYENA`, `METAGENE`, or `GenomeOcean` depending on `base_model_name` |
| `base_model_name` | HuggingFace model name, defaults to `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`  |
| `tokenizer_name` | Tokenizer model name, defaults to `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` |
| `tokenizer_kwargs` | Optional dict of tokenizer arguments |


### Training Parameters
| Parameter | Description |
|-----------|-------------|
| `train_iterable` | Whether to use `IterableDataset` for tokenizing training data with HF Trainer (defaults to `False`)|
| `num_rows_iterable` | Number of rows when using `IterableDataset` (default `None`) |
| `use_class_weights` | Whether to use class weights (stored in the `data_processor` file) for computing train/val loss (useful for imbalanced classification) |
| `train_batch_size` | Training batch size per GPU (default `16`) |
| `eval_batch_size` | Evaluation batch size per GPU (default `16`) |
| `epochs` | Maximum number of training epochs (default `3`) |
| `learning_rate` | Initial learning rate for AdamW optimizer (default `2e-5`) |
| `fp16` | Use 16-bit (mixed) precision training (`True`/`False`) |
| `bf16` | Use bfloat16 precision (preferred on newer GPUs) |
| `weight_decay` | Weight decay coefficient for regularization (e.g., `0.01`) |
| `warmup_ratio` | Fraction of training steps used for learning rate warmup (e.g., `0.05`) |
| `lr_scheduler_type` | Learning rate scheduler type (default `"cosine"`) |
| `gradient_accumulation_steps` | Steps to accumulate gradients before optimizer update (default `1`) |

### Validation Parameters
| Parameter | Description |
|-----------|-------------|
| `eval_accumulation_steps` | Number of steps before transferring predictions from GPU to CPU (helps avoid OOM errors) |
| `prediction_loss_only` | Whether to only output/save validation loss without individual predictions |

### Testing Parameters
| Parameter | Description |
|-----------|-------------|
| `predictions_batch` | Batch size for testing dataset results aggregation (default `None`) |
| `save_probabilities` | Whether to save per-class probabilities for test set (can be memory-intensive) |

### Model Selection & Early Stopping
| Parameter | Description |
|-----------|-------------|
| `multi_gpu_count` | Number of GPUs used for training using `acclerate` framework for multi-gpu training (default `1`) |
| `metric_for_best_model` | Metric used to determine the best model (default `"eval_loss"`) |
| `greater_is_better` | Whether a higher value of the metric indicates better performance (default `False` for loss) |
| `early_stopping_patience` | Number of epochs with no improvement before stopping training (default `3`) |

### Additional Training Parameters
Use these parameters if using additional training methods such as Parameter-Efficient Fine-Tuning (PEFT) methods to fine-tune models.

| Parameter | Description |
|-----------|-------------|
| `peft_method` | `"lora"`, `"ia3"`, or `None` |
| `lora_r` | Rank for LoRA if enabled |
| `randomization` | Whether to randomize model weights before training (`True`/`False`) |
| `freeze_layers_fraction` | Fraction of model layers to freeze during training (0.0–1.0) |


### Saving & Logging
| Parameter | Description |
|-----------|-------------|
| `experiment_name` | Name of directory for saving all resulting outputs such as model checkpoints, test results, etc. for this run (auto-generated if none specified) |
| `script_args_file` | File name for storing configuration arguments saved in the output directory |
| `epochs_trained_file` | File name for tracking the number of epochs trained (useful for pre-emptible training) |


---

### Configuration Tips

- You **must** define either a training dataset path or a testing dataset path in the config `yaml` to ensure that training or evaluation mode is run. 
- Make sure `experiment_dir` is writable (default is hardcoded to `~/analysis/analysis/experiment` in the class).
- If not explicitly specified, `experiment_name` defaults to a random set of 24 characters as a unique identifier of the experiment run.
- By default, all experiment runs are output to the directory `~/analysis/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/`.
- All logs from training runs are saved to a single, timestamped log file in the experiment's save directory `~/analysis/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/`.
- `testing_name` is name of subdirectory that stores results of a given test run, defaults to `test_results`. 
- Outputs of the experiment run include label predictions, all label probabilities, experimental run arguments, saved off data_processor `.pkl` file. All model checkpoints, model training history values and metric plot, and model performance on test dataset are stored in `~/analysis/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/` and subfolders within.
- The best model is saved off as `~/analysis/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/model.safetensors`.
- Model checkpoints and training history are stored in `~/analysis/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/train_history`.
- Model test results are stored in `~/analysis/analysis/experiment/runs/EXPERIMENT_NAME/MODEL_NAME/test_results`
- If path to training data is specified in `training_data`, then the script runs in `training` mode and trains on a subset of this data. If `validation_data` and/or `testing_data` are also specified, then best trained model will evaluate on the datasets specified by these path. Otherwise, the validation and/or testing dataset will be sampled from the training_data (80%/10%/10% split). Option for stratifying the train/val/test split is contained in the condfig parameter `stratify`, which must be a valid input data column. 
- If `training_data` is not specified, then the script is not in `training` mode and does train on any dataset. Instead, the top performing model from the specified `experiment_name` run is loaded in used to evaluate on the testing_data specified in `testing_data`. Note that `testing_data` must be specified in this case to ensure that eval is done on a valid test set.
- `tokenizer_kwargs` is a dict of optional keyword arguments passed into the tokenizer.
- List of possible `model_types`, `tokenizer_name`, and `base_model_name` are contained in `analysis.experiment.utils.constants`.
- If doing a multiclass classification, make sure `labels` is a list of labels to classify over. 
- Label encoding and dataset tokenization are cached, and tokenized datasets are saved for faster reloads when rerunning the same training script. If you specify `tokenized_training_data` , `tokenized_validation_data`  and/or `tokenized_testing_data` parameters in the config, then it will use the datasets corresponding to those filepaths (directories) explicitly. Otherwise, it will check in the default location for these datasets if they exist, and load those in. If those also dont exist, it tokenizes from scratch and saves off in the default location.
- If a model was trained in a previous run, subsequent runs will automatically resume from the last saved checkpoint—preemptibility is enabled to support seamless continuation of training.

### Generated Directory Structure

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
            └── train_DATE_TIME_.log
```

#### Obtaining Class Metrics
Evaluation metrics per individual class can be generated by running the script
`~/analysis/experiment/utils/metrics_generator.py`

Example usage: 
```
parent_dir = "/home/apluser/analysis/analysis/experiment/runs/bertax/full/"
generator = MetricsGenerator(None)
generator.process_multiple_models(parent_dir)
```

This will generate a `class_metrics.csv` file in each model output directory, containing the F1 score, precision, recall, and support for each class with at least one example in the test set.

Additional features of the class:
- Identifying the correlation between F1 score and support.
- Generating a confusion matrix.
- Calculating the Jaccard similarity between the predictions of each model in the parent_dir.

# Setup and run from Docker container

Before starting the container, you must have a preconfigured model directory on your host machine that contains the relevant data_processor file, tokenizer, and trained model weights. 
This directory should follow the structure below:
```
MODEL_DIR/
  └── base_model/
  └── data_processor/
  └── model/
```
This type of directory is available for download at [HuggingFace](https://huggingface.co/jhuapl-bio)

### Directory Details
- **`MODEL_DIR/base_model`**  
  Holds the base_model and tokenizer files necessary for preprocessing input sequences and loading fine-tuned model.  
- **`MODEL_DIR/data_processor`**
  Contains the data processor used to store and encode model inference labels.
- **`MODEL_DIR/model`**  
  Contains the trained model weight files (e.g., `model.safetensors`) that are loaded in for evaluating sequences.

### Build the Container
From the project root, build the image:
```
docker build --tag microbert .
```

### Run the Container
Run the container and mount your local `data/` directory into the container at `/analysis/data`:
```
docker run -d --rm \
  --name microbert \
  -p 3100:3100 \
  -v "$(pwd)/data:/analysis/data" \
  -e PYTHONPATH=/analysis \
  microbert-test
 ```
This ensures the container has access to the data directory while keeping your application code inside the image.

### Running with GPU Support
If your host has CUDA and the NVIDIA Container Toolkit installed, enable GPU usage with the flag `--gpus all`
See the following for instruction details:
- [CUDA Installation Guide (Linux)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)  
- [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)  

### Test Sequences
To run evaluate sequences from a running container:
```
docker exec -it microbert ./analysis/experiment/test_sequences.py -i INPUT_FASTA_FILE -o OUTPUT_JSON_PATH -d MODEL_DIR
```
e.g. 
```
docker exec -it microbert ./analysis/experiment/test_sequences.py -i data/input/test_sample_sub.fasta -o data/output/test.json -d data/LongSafari__hyenadna-large-1m-seqlen-hf
```