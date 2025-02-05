import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from roof_dataset import RoofDataset
from tqdm import tqdm
from roofsam.build_roofsam import build_roofsam_from_sam_vit_h_checkpoint
from sklearn.metrics import f1_score
import numpy as np


def set_requires_grad(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad


DATASET_ROOT = "dataset"

# Anzahl der Epochen festlegen
NUM_EPOCHS = 10

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

if __name__ == "__main__":
    train_dataset, test_dataset, num_classes, class_to_index, class_weights = (
        RoofDataset.get_train_test_split(DATASET_ROOT, num_sampled_points=4, min_samples_threshold=100)
    )

    index_to_class = {v: k for k, v in class_to_index.items()}

    train_dl = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=4,
        persistent_workers=True,
        shuffle=True,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=4,
        persistent_workers=True,
        shuffle=False,
    )  # No shuffle for validation

    # TODO: Remove MPS
    device = torch.device("mps")  # oder "cuda" für NVIDIA GPUs
    roofsam = build_roofsam_from_sam_vit_h_checkpoint(
        num_classes=num_classes, sam_checkpoint="sam_vit_h_4b8939.pth"
    ).to(device)

    # Gesamtes Modell in den Evaluationsmodus versetzen und einfrieren
    roofsam.eval()
    set_requires_grad(roofsam.parameters(), False)

    # Mask Decoder für das Training vorbereiten
    roofsam.mask_decoder.train()
    set_requires_grad(roofsam.mask_decoder.parameters(), True)

    # Adam-Optimierer für den Mask Decoder initialisieren
    optimizer = optim.Adam(roofsam.mask_decoder.parameters(), lr=1e-4)

    # Verlustfunktion definieren
    loss_ce = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_mf1 = 0.0
    for epoch in range(NUM_EPOCHS):
        roofsam.mask_decoder.train()  # Ensure training mode
        train_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
            embeddings = batch[0].to(device)
            point_coords = batch[1].to(device)
            point_labels = torch.ones((batch[1].shape[0], batch[1].shape[1])).to(device)
            targets = batch[2].unsqueeze(1).repeat((1, batch[1].shape[1])).to(device)

            # Vorwärtsdurchlauf
            class_logits = roofsam(
                {
                    "embeddings": embeddings,
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                }
            )

            # Verlust berechnen
            loss = loss_ce(class_logits, targets)

            # Gradienten zurücksetzen
            optimizer.zero_grad()

            # Rückwärtsdurchlauf
            loss.backward()

            # Parameter aktualisieren
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dl)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        roofsam.mask_decoder.eval()  # Set model to evaluation mode
        val_loss = 0.0

        # Lists to store all predictions and targets
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(
                test_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"
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
                class_logits = roofsam(
                    {
                        "embeddings": embeddings,
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                    }
                )

                # Compute loss for the batch
                loss = loss_ce(class_logits, targets)
                val_loss += loss.item()

                # Compute predictions (assuming the logits are for multiple classes)
                # Here, we assume class_logits has shape [batch_size, num_classes, num_points]
                preds = torch.argmax(
                    class_logits, dim=1
                )  # shape: [batch_size, num_points]

                # Flatten predictions and targets for metric computation
                preds_flat = preds.view(-1).cpu().numpy()
                targets_flat = targets.view(-1).cpu().numpy()

                all_preds.append(preds_flat)
                all_targets.append(targets_flat)

        # Compute the average validation loss over batches
        avg_val_loss = val_loss / len(test_dl)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")

        # Concatenate all batch predictions and targets into single arrays
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Compute F1 scores per class and a mean (macro) F1 score
        f1_per_class = f1_score(all_targets, all_preds, average=None)
        f1_mean = f1_score(all_targets, all_preds, average="macro")

        # Print F1 scores for each class
        for class_idx, f1 in enumerate(f1_per_class):
            print(f"F1 Score for class {index_to_class[class_idx]}: {f1:.4f}")
        print(f"Mean F1 Score (macro-average): {f1_mean:.4f}")

        # Checkpoint storing based on the largest mF1 score
        if f1_mean > best_mf1:
            best_mf1 = f1_mean
            checkpoint_path = os.path.join(
                checkpoint_dir, f"mask_decoder_epoch{epoch+1}_mf1_{f1_mean:.4f}.pt"
            )
            torch.save(roofsam.mask_decoder.state_dict(), checkpoint_path)
            print(f"New best model found and saved at {checkpoint_path}")
