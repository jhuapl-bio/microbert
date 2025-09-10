#!/usr/bin/env python3
# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

from argparse import ArgumentParser
import json

from math import ceil

from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
)
from safetensors.torch import load_file

from analysis.experiment.utils.data_processor import DataProcessor
from analysis.experiment.models.hierarchical_model import (
    HierarchicalClassificationModel,
)
from analysis.experiment.utils.test_utils import parse_input_fasta, normalize_label
from analysis.experiment.utils.train_logger import logger


def main(
    input_path: str,
    output_path: str,
    base_model_name: str = "LongSafari/hyenadna-large-1m-seqlen-hf",
    device: str = "cpu",
    batch_size: int = 256,
    top_k: int = 5,
    threshold: float = 0.2,
):

    """
    Runs batched inference on input FASTA/FASTQ sequences, saves the top predictions per label to JSON file.

    Args:
        input_path (str): Path to the input FASTA/FASTQ file (.fa/.fq, optionally .gz).
        output_path (str): Path to the output probabilities json file.
        base_model_name (str): Name of the fine-tuned genomic language model to use.
        device (str): either cpu or cuda device to run evaluations. Defaults to cpu
        batch_size (int): Number of sequences to process per batch, defaults to 256
        top_k (int): Number of top predictions to extract per label. Defaults to top 5.
        threshold (float): Only return predicted labels with probability > threshold

    Returns:
        Writes a list with predictions per sequence
    """
    # model specifications
    logger.info(f"model used: {base_model_name}")
    base_model_name = base_model_name.replace('/', '__')

    # data processor specifications
    data_processor_filename = 'data_processor.pkl'
    data_processor_dir = Path("data") / base_model_name / "data_processor"
    metadata_filename = 'metadata.json'
    metadata_path = Path("data") / base_model_name / "data_processor" / metadata_filename
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Access fields
    sequence_column = metadata["sequence_column"]
    labels = metadata["labels"]

    logger.info(f"Using data processor {data_processor_dir}/{data_processor_filename}")
    logger.info(f"sequence column: {sequence_column}")
    logger.info(f"labels: {labels}")
    logger.info(f"Using device: {device}")
    logger.info(f"batch size for evaluation: {batch_size}")
    logger.info(f"top k classes: {top_k}")
    logger.info(f"threshold for probabilities: {threshold}")

    # load data processor
    data_processor = DataProcessor(
        sequence_column=sequence_column,
        labels=labels,
        save_file=data_processor_filename,
    )
    data_processor.load_processor(data_processor_dir)
    
    num_labels = data_processor.num_labels
    class_weights = data_processor.class_weights

    # Load data tokenizer from local download
    local_base_model_dir = (
        Path("data") / base_model_name / "base_model"
    ).as_posix()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=local_base_model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    logger.info(f"Instantiated tokenizer for {base_model_name}")

    # Load safetensors weights
    trained_model_path = (
        Path("data") / base_model_name / "model" / "model.safetensors"
    ).as_posix()

    # TODO Fix head layer swap
    if "nucleotide-transformer-v2" in base_model_name:
        num_labels[3], num_labels[2] = num_labels[2], num_labels[3]
    model = HierarchicalClassificationModel(local_base_model_dir, num_labels, class_weights)
    state_dict = load_file(trained_model_path)
    logger.info(f"Loading model weights from {trained_model_path}...")
    model.load_state_dict(state_dict, strict=False)
    
    # move model to GPU if available
    model.to(device)
    
    input_sequences = parse_input_fasta(input_path)
    logger.info(f"Parsed input sequences at {input_path}")
    num_batches = ceil(len(input_sequences) / batch_size)
    logger.info(f"Processing {num_batches} batches of batch size {batch_size}")
    
    topk_all_sequences = []  # Store results across all batches
    
    for batch_idx in range(num_batches):
        logger.info(f"Processing batch {batch_idx + 1} of {num_batches}")
    
        batch_sequences = input_sequences[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    
        # Tokenize with padding and truncation
        tokenized_batch = tokenizer(
            batch_sequences,
            return_tensors="pt", # Return results as PyTorch tensors
            padding=True, # Pad sequences to the longest in the batch
            truncation=True # Truncate sequences longer than max length
        )
    
        # Move to same device as model
        tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
    
        with torch.no_grad():
            outputs = model(**tokenized_batch)
    
        # In-batch result collection
        batch_topk = []
    
        for i in range(len(batch_sequences)):
            instance_topk_label_probs = []
    
            for idx, col in enumerate(labels):
                logits = outputs['logits'][idx][i]  # [num_classes]
                top_k_col = min(top_k, logits.shape[-1])
                probs = F.softmax(logits, dim=-1).cpu()
    
                topk = torch.topk(probs, k=top_k_col, dim=-1)
                topk_indices = topk.indices
                topk_probs = topk.values
    
                topk_labels = data_processor.encoders[col].inverse_transform(topk_indices.numpy())

                label_probs = [
                    {
                        "label": normalize_label(label),
                        "probability": prob
                    }
                    for label, prob in zip(topk_labels, topk_probs.tolist())
                    if prob > threshold
                ]
    
                instance_topk_label_probs.append({
                    "label_column": col,
                    "topk_predictions": label_probs
                })
    
            batch_topk.append(instance_topk_label_probs)
    
        # Append this batch's results
        topk_all_sequences.extend(batch_topk)
    
    # return predictions as dict
    logger.info(f"Returning top k predictions for {len(topk_all_sequences)} sequences")

    # Convert to JSON string (pretty formatted)
    predictions_json = json.dumps(topk_all_sequences, indent=2)

    # Save JSON to file
    with open(output_path, "w") as f:
        f.write(predictions_json)

    logger.info(f"Predictions written to {output_path}")

    # Also return the JSON string (or dict if you prefer)
    return predictions_json

def parse_args():
    parser = ArgumentParser(description="Run batched inference using a trained gLM")

    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        default="data/input/test_sample_sub.fasta",
        help="Path to input FASTA file",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="data/output/test_sample_sub_results.json",
        help="Path to write JSON results",
    )

    parser.add_argument(
        "-m",
        "--base-model-name",
        type=str,
        default="LongSafari/hyenadna-large-1m-seqlen-hf",
        help="Base model to use",
    )

    parser.add_argument(
        "--use-gpu", action="store_true", help="Enable GPU if available"
    )

    parser.add_argument("-b", "--batch-size", type=int, default=256, help="Batch size")

    parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="Top-K predictions to keep"
    )

    parser.add_argument(
        "-t", "--threshold", type=float, default=0.2, help="Score threshold"
    )

    return parser

if __name__ == "__main__":
    args = parse_args().parse_args()
    # Define the device to run evaluations on
    logger.info(f"Using GPU if available: {args.use_gpu}")
    if bool(args.use_gpu):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu" # force CPU
    
    results = main(
        input_path=args.input_path,
        output_path=args.output_path,
        base_model_name=args.base_model_name,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    logger.info(results)
