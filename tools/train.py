import os
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from utils import majority_vote, set_requires_grad

from roofsam.datasets.alkis_roof_dataset import AlkisRoofDataset
from roofsam.build_roofsam import build_roofsam_from_sam_vit_h_checkpoint


def main(args):
    # Ensure the checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Get the train/test split from the dataset.
    # These parameters must match those used during training.
    train_dataset, test_dataset, num_classes, index_to_class, class_weights = (
        AlkisRoofDataset.get_train_test_split(
            args.dataset_root,
            num_sampled_points=args.num_sampled_points,
            min_samples_threshold=args.min_class_instances,
        )
    )

    # Create DataLoaders for training and testing
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=False,  # No shuffling for evaluation
    )

    # Set the device (e.g., MPS or CUDA)
    device = torch.device(args.device)

    # Build the model and load the SAM checkpoint
    model = build_roofsam_from_sam_vit_h_checkpoint(
        num_classes=num_classes, sam_checkpoint=args.sam_checkpoint
    ).to(device)

    # Freeze the entire model and set it to evaluation mode
    model.eval()
    set_requires_grad(model.parameters(), False)

    # Prepare the mask decoder for training
    model.mask_decoder.train()
    set_requires_grad(model.mask_decoder.parameters(), True)

    # Initialize the Adam optimizer for the mask decoder
    optimizer = optim.Adam(model.mask_decoder.parameters(), lr=args.learning_rate)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_macro_f1 = 0.0
    for epoch in range(args.num_epochs):
        # Set mask decoder to training mode
        model.mask_decoder.train()
        training_loss = 0.0

        # Training loop
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Training"
        ):
            embeddings = batch[0].to(device)
            point_coords = batch[1].to(device)
            # Create dummy point labels (the model expects them)
            point_labels = torch.ones((batch[1].shape[0], batch[1].shape[1])).to(device)
            # Repeat the target for all points in the sample
            targets = batch[2].unsqueeze(1).repeat((1, batch[1].shape[1])).to(device)

            # Forward pass
            logits = model(
                {
                    "embeddings": embeddings,
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                }
            )

            # Compute the loss
            loss = loss_fn(logits, targets)

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            training_loss += loss.item()

        avg_training_loss = training_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {avg_training_loss:.4f}"
        )

        # Validation loop
        model.mask_decoder.eval()  # Set model to evaluation mode
        validation_loss = 0.0

        # Lists to store majority predictions and targets per sample
        all_majority_preds = []
        all_majority_targets = []

        with torch.no_grad():
            for batch in tqdm(
                test_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Validation"
            ):
                embeddings = batch[0].to(device)
                point_coords = batch[1].to(device)
                point_labels = torch.ones((batch[1].shape[0], batch[1].shape[1])).to(
                    device
                )
                targets = (
                    batch[2].unsqueeze(1).repeat((1, batch[1].shape[1])).to(device)
                )

                # Forward pass: get class logits
                logits = model(
                    {
                        "embeddings": embeddings,
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                    }
                )

                # Compute loss for the batch
                loss = loss_fn(logits, targets)
                validation_loss += loss.item()

                # Compute predictions. Assuming logits shape is
                # [batch_size, num_classes, num_points]
                preds = torch.argmax(logits, dim=1)  # shape: [batch_size, num_points]

                # Compute majority vote per sample
                majority_preds = majority_vote(preds)  # shape: [batch_size]
                # Since all points in a sample have the same target, use the first point's target
                majority_targets = targets[:, 0]  # shape: [batch_size]

                all_majority_preds.append(majority_preds.cpu().numpy())
                all_majority_targets.append(majority_targets.cpu().numpy())

        avg_validation_loss = validation_loss / len(test_loader)
        print(
            f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {avg_validation_loss:.4f}"
        )

        # Concatenate all majority predictions and targets from all batches
        all_majority_preds = np.concatenate(all_majority_preds)
        all_majority_targets = np.concatenate(all_majority_targets)

        # Compute F1 scores (per class and macro-average)
        f1_per_class = f1_score(all_majority_targets, all_majority_preds, average=None)
        macro_f1 = f1_score(all_majority_targets, all_majority_preds, average="macro")

        # Print per-class F1 scores
        for class_idx, f1 in enumerate(f1_per_class):
            print(f"F1 Score for class {index_to_class[class_idx]}: {f1:.4f}")
        print(f"Mean F1 Score (macro-average): {macro_f1:.4f}")

        # Save checkpoint if the macro-average F1 score improves
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"mask_decoder_epoch{epoch+1}_mf1_{macro_f1:.4f}.pt",
            )
            torch.save(model.mask_decoder.state_dict(), checkpoint_path)
            print(f"New best model found and saved at {checkpoint_path}")

if __name__ == "__main__":
    # Define command-line arguments for all the constants
    parser = argparse.ArgumentParser(description="Train the RoofSAM decoder.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--num_sampled_points",
        type=int,
        default=4,
        help="Number of points to sample per instance.",
    )
    parser.add_argument(
        "--min_class_instances",
        type=int,
        default=100,
        help="Minimum number of instances per class required.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for the DataLoader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for the DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (e.g., 'mps' or 'cuda').",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file.",
    )
    my_args = parser.parse_args()
    main(my_args)
