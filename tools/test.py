import os
import sys
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from roofsam.datasets.alkis_roof_dataset import AlkisRoofDataset
from roofsam.build_roofsam import build_roofsam_from_sam_vit_h_checkpoint

# Set the dataset root and device
DATASET_ROOT = "dataset"
device = torch.device("mps")  # or "cuda" if using NVIDIA GPUs

def majority_vote(preds: torch.Tensor) -> torch.Tensor:
    """
    Computes the majority vote for each sample (each row of predictions).
    Since torch.mode is not supported on MPS, we use NumPy.
    """
    preds_cpu = preds.cpu().numpy()  # shape: [batch_size, num_points]
    majority = []
    for sample in preds_cpu:
        counts = np.bincount(sample)
        majority.append(np.argmax(counts))
    # Return a tensor with the majority vote for each sample on the original device
    return torch.tensor(majority, device=preds.device)

def main(checkpoint_path):
    # Create the dataset and test split.
    # These parameters must match those used during training.
    _, test_dataset, num_classes, index_to_class, class_weights = (
        AlkisRoofDataset.get_train_test_split(
            DATASET_ROOT, num_sampled_points=4, min_samples_threshold=100
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=4,
        persistent_workers=True,
        shuffle=False,  # no shuffling for evaluation
    )

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    # Build the model. Ensure that the num_classes and checkpoint used are consistent.
    model = build_roofsam_from_sam_vit_h_checkpoint(
        num_classes=num_classes,
        sam_checkpoint="sam_vit_h_4b8939.pth",
        roof_sam_mask_decoder_checkpoint=checkpoint_path,
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
            # Create dummy point labels (the model expects them)
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
    if len(sys.argv) < 2:
        print("Usage: python test.py <checkpoint_path>")
        sys.exit(1)
    checkpoint_path = sys.argv[1]
    main(checkpoint_path)
