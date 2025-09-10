# train_mlp.py
# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

import os
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from analysis.experiment.utils.data_processor import DataProcessor
from analysis.experiment.utils.config import CONFIG_DIR, Config
from analysis.experiment.utils.embeddings_utils import read_group_in_batches
from analysis.experiment.utils.train_logger import add_file_handler, logger
from analysis.experiment.utils.train_utils import multiclass_metrics


# ------------------------------------------------------------------
# 1) PyTorch Dataset
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 2) Model Definition (Dynamic MLP with BN + Dropout)
# ------------------------------------------------------------------
class ChainHierModel(nn.Module):
    """
    For hierarchical classification superkingdom -> phylum -> genus,
    allowing a dynamic # of layers in the shared MLP via config.num_layers.
    """

    def __init__(
        self,
        embed_dim,
        num_superkingdoms,
        num_phyla,
        num_genera,
        num_layers=2,
        layer_size=256,
        dropout_prob=0.5,
    ):
        super().__init__()

        # Build a dynamic list of (Linear -> BN -> ReLU -> Dropout) blocks
        self.shared_layers = nn.ModuleList()
        input_dim = embed_dim
        for _ in range(num_layers):
            self.shared_layers.append(nn.Linear(input_dim, layer_size))
            self.shared_layers.append(nn.BatchNorm1d(layer_size))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(dropout_prob))
            input_dim = layer_size

        # Heads:
        # superkingdom sees the final hidden dim
        self.sk_head = nn.Linear(layer_size, num_superkingdoms)

        # phylum sees hidden_dim + num_superkingdoms (one-hot)
        self.ph_head = nn.Linear(layer_size + num_superkingdoms, num_phyla)

        # genus sees hidden_dim + num_phyla (one-hot)
        self.ge_head = nn.Linear(layer_size + num_phyla, num_genera)

        self.num_superkingdoms = num_superkingdoms
        self.num_phyla = num_phyla
        self.num_genera = num_genera

    def forward(
        self,
        embeddings,
        parent_labels=None,
        mode="train",
    ):
        """
        Shared MLP pass => chain logic for SK -> PH -> GE
        """
        x = embeddings
        # Pass through each 4-layer block: (Linear, BN, ReLU, Dropout)
        # If you have 2 layers => len(self.shared_layers) = 8
        for i in range(0, len(self.shared_layers), 4):
            x = self.shared_layers[i](x)  # Linear
            x = self.shared_layers[i + 1](x)  # BN
            x = self.shared_layers[i + 2](x)  # ReLU
            x = self.shared_layers[i + 3](x)  # Dropout

        # shape: [bs, layer_size]

        # 1) superkingdom
        logits_sk = self.sk_head(x)

        # Decide how to feed phylum
        if mode == "train" and parent_labels is not None:
            # teacher forcing => ground truth superkingdom
            sk_label = parent_labels[:, 0]
        else:
            sk_label = logits_sk.argmax(dim=1)
        sk_label_1hot = torch.nn.functional.one_hot(
            sk_label, num_classes=self.num_superkingdoms
        ).float()

        ph_input = torch.cat([x, sk_label_1hot], dim=1)
        logits_ph = self.ph_head(ph_input)

        # 2) genus
        if mode == "train" and parent_labels is not None:
            ph_label = parent_labels[:, 1]
        else:
            ph_label = logits_ph.argmax(dim=1)
        ph_label_1hot = torch.nn.functional.one_hot(
            ph_label, num_classes=self.num_phyla
        ).float()

        ge_input = torch.cat([x, ph_label_1hot], dim=1)
        logits_ge = self.ge_head(ge_input)

        return logits_sk, logits_ph, logits_ge


class HierarchicalMLP(nn.Module):
    """
    Multi-head MLP for hierarchical classification.
    Includes BatchNorm and Dropout in the shared MLP backbone.
    """

    def __init__(
        self,
        input_dim,
        output_dims,
        hidden_dim=256,
        dropout_prob=0.3,
    ):
        """
        Args:
          input_dim: size of the input embedding vector
          output_dims: list of integer sizes for each classification head
                       e.g. [num_classes_rank1, num_classes_rank2, ...]
          hidden_dim: hidden dimension for the shared MLP backbone
          dropout_prob: dropout probability to use in the backbone
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        # Build a separate head (Linear) for each rank
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, out_dim) for out_dim in output_dims]
        )

    def forward(self, x):
        # Shared backbone: FC -> BN -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Return a list of logits for each head
        return [head(x) for head in self.heads]


# ------------------------------------------------------------------
# 3) Training & Evaluation Helpers
# ------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, device="cuda"):
    """Trains for one epoch, returning the average training loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    for embeddings, label_mat in tqdm(dataloader, desc="Train", leave=False):
        embeddings = embeddings.to(device)
        label_mat = label_mat.to(device)

        optimizer.zero_grad()
        logits_sk, logits_ph, logits_ge = model(
            embeddings, parent_labels=label_mat, mode="train"
        )

        loss_sk = criterion(logits_sk, label_mat[:, 0])  # superkingdom
        loss_ph = criterion(logits_ph, label_mat[:, 1])  # phylum
        loss_ge = criterion(logits_ge, label_mat[:, 2])  # genus
        loss = loss_sk + loss_ph + loss_ge

        loss.backward()
        optimizer.step()

        bs = embeddings.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples


def train_one_epoch_unconditioned(model, loader, optimizer, device="cuda"):
    """Train for one epoch on the given data loader."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    for xb, yb in tqdm(loader, desc="Training", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)  # shape (batch_size, num_ranks)
        optimizer.zero_grad()

        logits_list = model(xb)  # list of [batch_size x out_dim]
        # If we have multiple ranks: yb[:, i] for each rank i
        # Sum up CrossEntropy across all heads
        loss = 0.0
        for i, logits in enumerate(logits_list):
            loss += criterion(logits, yb[:, i])

        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples


def compute_validation_loss(model, dataloader, device="cuda", unconditioned=False):
    """Compute average validation loss (teacher forcing approach)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for embeddings, label_mat in tqdm(dataloader, desc="Val Loss", leave=False):
            embeddings = embeddings.to(device)
            label_mat = label_mat.to(device)

            # same approach as training, but no backward pass
            logits_sk, logits_ph, logits_ge = (
                model(embeddings, parent_labels=label_mat, mode="train")
                if not unconditioned
                else model(embeddings)
            )
            # Summation of cross-entropy
            loss_sk = criterion(logits_sk, label_mat[:, 0])
            loss_ph = criterion(logits_ph, label_mat[:, 1])
            loss_ge = criterion(logits_ge, label_mat[:, 2])
            loss = loss_sk + loss_ph + loss_ge

            bs = embeddings.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

    return total_loss / total_samples


def evaluate(model, dataloader, device="cuda"):
    """
    Evaluate in "eval" mode (sequential chain from parent's predicted label).
    Returns (predictions, labels) for each rank.
    """
    model.eval()
    all_preds_sk = []
    all_preds_ph = []
    all_preds_ge = []
    all_labels_sk = []
    all_labels_ph = []
    all_labels_ge = []

    with torch.no_grad():
        for embeddings, label_mat in tqdm(dataloader, desc="Eval", leave=False):
            embeddings = embeddings.to(device)
            label_mat = label_mat.to(device)

            logits_sk, logits_ph, logits_ge = model(
                embeddings, parent_labels=None, mode="eval"
            )

            preds_sk = logits_sk.argmax(dim=1)
            preds_ph = logits_ph.argmax(dim=1)
            preds_ge = logits_ge.argmax(dim=1)

            all_preds_sk.append(preds_sk.cpu())
            all_preds_ph.append(preds_ph.cpu())
            all_preds_ge.append(preds_ge.cpu())

            all_labels_sk.append(label_mat[:, 0].cpu())
            all_labels_ph.append(label_mat[:, 1].cpu())
            all_labels_ge.append(label_mat[:, 2].cpu())

    all_preds_sk = torch.cat(all_preds_sk).numpy()
    all_preds_ph = torch.cat(all_preds_ph).numpy()
    all_preds_ge = torch.cat(all_preds_ge).numpy()

    all_labels_sk = torch.cat(all_labels_sk).numpy()
    all_labels_ph = torch.cat(all_labels_ph).numpy()
    all_labels_ge = torch.cat(all_labels_ge).numpy()

    return (all_preds_sk, all_preds_ph, all_preds_ge), (
        all_labels_sk,
        all_labels_ph,
        all_labels_ge,
    )


def evaluate_unconditioned(model, loader, device="cuda"):
    """
    Run inference on the dataset, collect predictions.
    Returns:
      all_preds: list of (num_ranks) torch tensors of shape [N]
      all_labels: shape [N, num_ranks]
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Evaluating", leave=False):
            xb = xb.to(device)
            logits_list = model(xb)  # list of [batch_size x out_dim] for each rank

            # Argmax for each rank -> shape (batch_size)
            preds = [logits.argmax(dim=1).cpu() for logits in logits_list]

            if not all_preds:
                # Initialize list of Tensors for each rank
                all_preds = [p.clone() for p in preds]
            else:
                for i, p in enumerate(preds):
                    all_preds[i] = torch.cat((all_preds[i], p), dim=0)
            all_labels.append(yb)  # shape (batch_size, num_ranks)

    all_labels = torch.cat(all_labels, dim=0).numpy()  # shape [N, num_ranks]
    # all_preds is a list, one for each rank
    # stack them dimension 1 => shape [N, num_ranks]
    all_preds_stacked = torch.stack(all_preds, dim=1).numpy()
    return (
        all_preds_stacked[:, 0],
        all_preds_stacked[:, 1],
        all_preds_stacked[:, 2],
    ), (all_labels[:, 0], all_labels[:, 1], all_labels[:, 2])


def compute_rank_metrics(preds, labs, encoder):
    """Compute metrics (macro-F1, accuracy, etc.) for a single rank."""
    pred_text = encoder.inverse_transform(preds.astype(int))
    lab_text = encoder.inverse_transform(labs.astype(int))
    return multiclass_metrics(lab_text, pred_text)


# ------------------------------------------------------------------
# 4) Main Script (Train w/Early Stopping on Val Loss, Save Best by Genus Macro-F1)
# ------------------------------------------------------------------
def main(config: Config, data_loaders=None, unconditioned=False):
    config.unconditioned = unconditioned

    # ---------------------------
    # 1. LOGGING SETUP
    # ---------------------------
    filename = "chain_hier_training"
    add_file_handler(logger, config.save_dir, filename)
    logger.info(
        f"Logger {filename} initialized. Logs will be saved to {config.save_dir}"
    )

    start_time = time.time()  # start timing the experiment
    if data_loaders:
        train_loader, val_loader, test_loader = data_loaders
    else:

        # ---------------------------
        # 3. LOAD EMBEDDINGS (TRAIN, VAL, TEST)
        # ---------------------------
        embeddings_save_path = config.embeddings_checkpoint_h5
        logger.info(f"Fetching embeddings from {embeddings_save_path}")

        train_embeddings, train_labels = read_group_in_batches(
            embeddings_save_path, "train"
        )
        val_embeddings, val_labels = read_group_in_batches(embeddings_save_path, "val")
        test_embeddings, test_labels = read_group_in_batches(
            embeddings_save_path, "test"
        )

        train_labels = np.asarray(train_labels)
        val_labels = np.asarray(val_labels)
        test_labels = np.asarray(test_labels)

        # ---------------------------
        # 4. BUILD DATASETS & LOADERS
        # ---------------------------
        logger.info(f"Batch size: {config.train_batch_size}")
        train_ds = HierarchicalDataset(train_embeddings, train_labels)
        val_ds = HierarchicalDataset(val_embeddings, val_labels)
        test_ds = HierarchicalDataset(test_embeddings, test_labels)

        train_loader = DataLoader(
            train_ds, batch_size=config.train_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.train_batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=config.train_batch_size, shuffle=False
        )

    # ---------------------------
    # 2. LOAD DATA PROCESSOR
    # ---------------------------
    data_processor = DataProcessor(
        sequence_column=config.sequence_column,
        labels=config.labels,
        save_file=config.data_processor_filename,
    )
    # Must have a data processor already fit ina valid directory specified in config
    data_processor.load_processor(config.data_processor_path)

    ranks = data_processor.labels  # e.g. ["superkingdom", "phylum", "genus"]
    logger.info(f"Taxonomic ranks: {ranks}")
    # ---------------------------
    # 5. BUILD THE MODEL
    #    We read from config.num_layers, config.layer_size, config.learning_rate
    # ---------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    num_superkingdoms = len(data_processor.encoders[ranks[0]].classes_)
    num_phyla = len(data_processor.encoders[ranks[1]].classes_)
    num_genera = len(data_processor.encoders[ranks[2]].classes_)

    example_embedding, _ = read_group_in_batches(
        config.embeddings_checkpoint_h5, "train", sample_size=1, batch_size=1
    )
    embed_dim = example_embedding.shape[1]

    if config.unconditioned:
        # Unconditioned MLP model (no chain logic)
        model = HierarchicalMLP(
            input_dim=embed_dim,
            output_dims=[num_superkingdoms, num_phyla, num_genera],
            hidden_dim=config.layer_size,
            dropout_prob=0.5,
        ).to(device)
    else:
        model = ChainHierModel(
            embed_dim=embed_dim,
            num_superkingdoms=num_superkingdoms,
            num_phyla=num_phyla,
            num_genera=num_genera,
            num_layers=config.num_layers,  # dynamic
            layer_size=config.layer_size,  # dynamic
            dropout_prob=0.5,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # We'll store checkpoints in this subfolder
    conditioned_dir = os.path.join(
        config.test_metrics_dir,
        "conditioned_mlp_dynamic" if not config.unconditioned else "unconditioned_mlp",
    )
    os.makedirs(conditioned_dir, exist_ok=True)

    # ---------------------------
    # 6. TRAINING LOOP (EARLY STOP ON VAL LOSS) + BEST EPOCH BY GENUS MACRO-F1
    # ---------------------------
    num_epochs = config.num_epochs if hasattr(config, "num_epochs") else 10
    early_stopping_patience = (
        config.early_stopping_patience
        if hasattr(config, "early_stopping_patience")
        else 3
    )
    logger.info(
        f"Starting training for {num_epochs} epochs with early stopping (patience={early_stopping_patience})"
    )

    best_epoch = 0
    best_macro_f1 = 0.0
    best_ckpt_path = None

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

        # 1) Training
        train_loss = (
            train_one_epoch(model, train_loader, optimizer, device=device)
            if not config.unconditioned
            else train_one_epoch_unconditioned(
                model, train_loader, optimizer, device=device
            )
        )
        logger.info(f"Train Loss: {train_loss:.4f}")

        # 2) Compute validation loss (for early stopping)
        val_loss = compute_validation_loss(
            model, val_loader, device=device, unconditioned=config.unconditioned
        )
        logger.info(f"Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"No improvement in val loss. Patience counter = {patience_counter}"
            )
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # 3) Evaluate on val set in "eval" mode to track genus macro-F1
        #    (You could skip this step if you only want to rely on val loss for early stopping,
        #     but here we also track which epoch gave the best genus macro-F1.)
        (pred_sk, pred_ph, pred_ge), (lab_sk, lab_ph, lab_ge) = (
            evaluate(model, val_loader, device=device)
            if not config.unconditioned
            else evaluate_unconditioned(model, val_loader, device=device)
        )

        # Build per-rank metrics
        epoch_metrics_dict = {}
        predictions_dict = {
            ranks[0]: (pred_sk, lab_sk),
            ranks[1]: (pred_ph, lab_ph),
            ranks[2]: (pred_ge, lab_ge),
        }
        for rank_name, (preds, labs) in predictions_dict.items():
            encoder = data_processor.encoders[rank_name]
            this_rank_metrics = compute_rank_metrics(preds, labs, encoder)
            epoch_metrics_dict[rank_name] = this_rank_metrics

        genus_macro_f1 = epoch_metrics_dict[ranks[2]].get("metric_f1_macro", 0.0)
        logger.info(f"Epoch {epoch+1} Genus Macro-F1 = {genus_macro_f1:.4f}")

        # Save the best model checkpoint if genus macro-F1 improved
        if genus_macro_f1 > best_macro_f1:
            best_macro_f1 = genus_macro_f1
            best_epoch = epoch + 1
            new_ckpt_path = os.path.join(conditioned_dir, "best_model.pt")

            # Remove old best checkpoint if needed
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                },
                new_ckpt_path,
            )
            best_ckpt_path = new_ckpt_path

            logger.info(
                f"New best checkpoint (epoch={best_epoch}) saved. Macro-F1={best_macro_f1:.4f}"
            )

    # ---------------------------
    # 7. Final Test Evaluation with the best model (by genus Macro-F1)
    # ---------------------------
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        logger.info(f"Loading best model from {best_ckpt_path} (epoch={best_epoch})")
        checkpoint = torch.load(best_ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning(
            "No best checkpoint found (never improved?). Using current model for final test eval."
        )

    logger.info("Evaluating final/best model on test set...")
    (pred_sk, pred_ph, pred_ge), (lab_sk, lab_ph, lab_ge) = (
        evaluate(model, test_loader, device=device)
        if not config.unconditioned
        else evaluate_unconditioned(model, test_loader, device=device)
    )

    # Build final metrics
    final_metrics = {}
    predictions_dict = {
        ranks[0]: (pred_sk, lab_sk),
        ranks[1]: (pred_ph, lab_ph),
        ranks[2]: (pred_ge, lab_ge),
    }
    for rank_name, (preds, labs) in predictions_dict.items():
        encoder = data_processor.encoders[rank_name]
        rank_m = compute_rank_metrics(preds, labs, encoder)
        final_metrics[rank_name] = rank_m

    # Summarize & Save
    def pretty_metric_name(s):
        return (
            s.replace("metric", "")
            .replace("_", " ")
            .title()
            .replace("Macro", "(Macro)")
            .replace("Micro", "(Micro)")
            .replace("Weighted", "(Weighted)")
            .replace("Mrr", "MRR")
        )

    cleaned_metrics = {
        rk: {pretty_metric_name(k): f"{v:.4f}" for k, v in md.items()}
        for rk, md in final_metrics.items()
    }
    df_metrics = pd.DataFrame.from_dict(cleaned_metrics, orient="index")

    # Add overall info to the DataFrame
    end_time = time.time()
    experiment_time = end_time - start_time
    df_metrics.loc["Overall", "Best Epoch"] = best_epoch
    df_metrics.loc["Overall", "Best F1 (Macro)"] = f"{best_macro_f1:.4f}"
    df_metrics.loc["Overall", "Experiment Time (s)"] = f"{experiment_time:.2f}"

    summary_path = os.path.join(conditioned_dir, "test_metrics_summary.csv")
    df_metrics.to_csv(summary_path)
    logger.info(f"Saved final summary metrics to {summary_path}")

    logger.info("\n=== Final Metrics ===")
    for idx, row in df_metrics.iterrows():
        logger.info(f"{idx}: {dict(row)}")

    logger.info("Chained hierarchical classification complete!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        default="bertax/sample/train_embeddings.yaml",
        nargs="?",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config_file = args.config_file
    config_path = os.path.join(CONFIG_DIR, config_file)
    logger.info(f"Using configuration file: {config_path}")

    config = Config(config_path)
    # 1) For example, config.num_layers, config.layer_size, config.train_batch_size, config.learning_rate
    #    might come from the YAML or you can override them here if needed:
    # config.num_layers = 4
    # config.layer_size = 512
    # config.train_batch_size = 64
    # config.learning_rate = 1e-4
    # config.early_stopping_patience = 3

    # Log config attributes
    for key, val in vars(config).items():
        logger.info(f"{key}: {val}")

    main(config)
