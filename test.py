import os
import sys
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from roof_dataset import RoofDataset
from roofsam.build_roofsam import build_roofsam_from_sam_vit_h_checkpoint

# Set the dataset root and device
DATASET_ROOT = "dataset"
device = torch.device("mps")  # or "cuda" if using NVIDIA GPUs


def main(checkpoint_path):
    # Create the dataset and test split.
    # These parameters must match those used during training.
    _, test_dataset, num_classes, class_to_index, class_weights = (
        RoofDataset.get_train_test_split(
            DATASET_ROOT, num_sampled_points=4, min_samples_threshold=100
        )
    )

    # Create a mapping from index to class name for printing F1 scores.
    index_to_class = {v: k for k, v in class_to_index.items()}

    test_dl = DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=4,
        persistent_workers=True,
        shuffle=False,  # no shuffling for evaluation
    )

    # Load the trained mask decoder weights from the checkpoint.
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    # Build the model. Ensure that the num_classes and checkpoint used are consistent.
    roofsam = build_roofsam_from_sam_vit_h_checkpoint(
        num_classes=num_classes,
        sam_checkpoint="sam_vit_h_4b8939.pth",
        roof_sam_mask_decoder_checkpoint=checkpoint_path,
    ).to(device)

    # Set the model to evaluation mode.
    roofsam.eval()
    roofsam.mask_decoder.eval()

    # Define the loss function (optional: if you want to report test loss)
    loss_ce = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Variables to accumulate loss and predictions.
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Testing"):
            embeddings = batch[0].to(device)
            point_coords = batch[1].to(device)
            # Create dummy point labels (the model expects them)
            point_labels = torch.ones((batch[1].shape[0], batch[1].shape[1])).to(device)
            targets = batch[2].unsqueeze(1).repeat((1, batch[1].shape[1])).to(device)

            # Forward pass through the model.
            class_logits = roofsam(
                {
                    "embeddings": embeddings,
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                }
            )

            # Compute loss.
            loss = loss_ce(class_logits, targets)
            val_loss += loss.item()

            # Get predictions. (Assuming logits shape [batch_size, num_classes, num_points])
            preds = torch.argmax(class_logits, dim=1)  # shape: [batch_size, num_points]

            # Flatten predictions and targets for F1 computation.
            preds_flat = preds.view(-1).cpu().numpy()
            targets_flat = targets.view(-1).cpu().numpy()

            all_preds.append(preds_flat)
            all_targets.append(targets_flat)

    # Compute average loss.
    avg_val_loss = val_loss / len(test_dl)
    print(f"\nTest Loss: {avg_val_loss:.4f}")

    # Concatenate all predictions and targets.
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute F1 scores.
    f1_per_class = f1_score(all_targets, all_preds, average=None)
    f1_mean = f1_score(all_targets, all_preds, average="macro")

    print("\nPer-class F1 scores:")
    for class_idx, f1 in enumerate(f1_per_class):
        # Use index_to_class to print a descriptive class name.
        class_name = index_to_class.get(class_idx, f"Class {class_idx}")
        print(f"  {class_name}: {f1:.4f}")
    print(f"\nMean F1 Score (macro-average): {f1_mean:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <checkpoint_path>")
        sys.exit(1)
    checkpoint_path = sys.argv[1]
    main(checkpoint_path)
