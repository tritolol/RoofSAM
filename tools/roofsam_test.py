#!/usr/bin/env python3
import os
import sys
import argparse

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from roofsam.utils import majority_vote

from roofsam.datasets.alkis_roof_dataset import AlkisRoofDataset
from roofsam.build_roofsam import build_roofsam_from_sam_vit_h_checkpoint


def main(args):
    # Set the device using the command-line argument
    device = torch.device(args.device)

    # Create the dataset and test split.
    # These parameters must match those used during training.
    _, test_dataset, num_classes, index_to_class, class_weights = (
        AlkisRoofDataset.get_train_test_split(
            args.dataset_root,
            num_sampled_points=args.num_sampled_points,
            min_samples_threshold=args.min_samples_threshold,
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=False,  # no shuffling for evaluation
    )

    # Check if the checkpoint file exists.
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)

    # Build the model.
    model = build_roofsam_from_sam_vit_h_checkpoint(
        num_classes=num_classes,
        sam_checkpoint=args.sam_checkpoint,
        roof_sam_mask_decoder_checkpoint=args.checkpoint_path,
    ).to(device)

    # Set the model to evaluation mode.
    model.eval()
    model.mask_decoder.eval()

    # Define the loss function (optional: if you want to report test loss)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Variables to accumulate loss and predictions.
    total_loss = 0.0
    all_majority_preds = []
    all_majority_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            embeddings = batch[0].to(device)
            point_coords = batch[1].to(device)
            # Create positive point labels
            point_labels = torch.ones((batch[1].shape[0], batch[1].shape[1])).to(device)
            targets = batch[2].unsqueeze(1).repeat((1, batch[1].shape[1])).to(device)

            # Forward pass through the model.
            logits = model(
                {
                    "embeddings": embeddings,
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                }
            )

            # Compute loss.
            loss = loss_fn(logits, targets)
            total_loss += loss.item()

            # Get predictions. (Assuming logits shape [batch_size, num_classes, num_points])
            preds = torch.argmax(logits, dim=1)  # shape: [batch_size, num_points]

            # Compute majority vote per sample.
            majority_preds = majority_vote(preds)  # shape: [batch_size]
            # Since all points have the same target value, use the first entry of each row.
            majority_targets = targets[:, 0]  # shape: [batch_size]

            all_majority_preds.append(majority_preds.cpu().numpy())
            all_majority_targets.append(majority_targets.cpu().numpy())

    # Compute average loss.
    avg_loss = total_loss / len(test_loader)
    print(f"\nTest Loss: {avg_loss:.4f}")

    # Concatenate all predictions and targets.
    all_majority_preds = np.concatenate(all_majority_preds)
    all_majority_targets = np.concatenate(all_majority_targets)

    # Compute F1 scores.
    f1_per_class = f1_score(all_majority_targets, all_majority_preds, average=None)
    f1_mean = f1_score(all_majority_targets, all_majority_preds, average="macro")

    print("\nPer-class F1 scores:")
    for class_idx, f1 in enumerate(f1_per_class):
        class_name = index_to_class.get(class_idx, f"Class {class_idx}")
        print(f"  {class_name}: {f1:.4f}")
    print(f"\nMean F1 Score (macro-average): {f1_mean:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the RoofSAM mask decoder model using a specified checkpoint."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the mask decoder checkpoint file.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'mps' or 'cuda').",
    )
    parser.add_argument(
        "--num_sampled_points",
        type=int,
        default=4,
        help="Number of sampled points to use.",
    )
    parser.add_argument(
        "--min_samples_threshold",
        type=int,
        default=100,
        help="Minimum number of samples required per class.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for testing.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for the DataLoader.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file.",
    )

    my_args = parser.parse_args()
    main(my_args)
