import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

# Import SAM modules
from segment_anything.build_sam import build_sam_vit_h
from segment_anything.predictor import SamPredictor


def download_checkpoint_if_needed(checkpoint):
    if not os.path.exists(checkpoint):
        print(f"Downloading checkpoint {checkpoint}...")
        os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/{checkpoint}")


def process_images_for_device(image_list, device_id, dop_path, emb_path, checkpoint):
    """
    Loads the SAM model onto the given CUDA device and processes all images in image_list.
    Saves each image's embedding to emb_path.
    """
    device = f"cuda:{device_id}"
    print(f"[Device {device}] Loading SAM model from {checkpoint}...")
    sam = build_sam_vit_h(checkpoint).to(device=device)
    predictor = SamPredictor(sam)

    for image in tqdm(image_list, desc=f"GPU {device_id}", position=device_id):
        # Create target embedding path
        basename, _ = os.path.splitext(image)
        target_path = os.path.join(emb_path, basename + ".pt")
        if os.path.exists(target_path):
            continue

        # Open image and run SAM to get the embedding
        image_path = os.path.join(dop_path, image)
        with Image.open(image_path) as im:
            im_array = np.array(im)
            predictor.set_image(im_array)
            emb = predictor.get_image_embedding()
            torch.save(emb, target_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process images using SAM with user-specified CUDA devices, dataset root, and checkpoint."
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default="all",
        help='Comma-separated list of CUDA device IDs to use (e.g. "0,1") or "all" to use all available devices.',
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset",
        help="Path to the dataset root directory. Assumes dop_images folder within.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="sam_vit_h_4b8939.pth",
        help="Path to the SAM checkpoint file. Defaults to the ViT-H model checkpoint (sam_vit_h_4b8939.pth).",
    )
    args = parser.parse_args()

    # Download the checkpoint if it does not exist.
    download_checkpoint_if_needed(args.checkpoint)

    # Set dataset paths based on the provided dataset root
    DATASET_ROOT = args.dataset_root
    DOP_PATH = os.path.join(DATASET_ROOT, "dop_images")
    EMB_PATH = os.path.join(DATASET_ROOT, "dop_embeddings")
    os.makedirs(EMB_PATH, exist_ok=True)

    total_devices = torch.cuda.device_count()
    if total_devices == 0:
        raise RuntimeError(
            "No CUDA devices found. This script requires at least one CUDA device."
        )
    print(f"Found {total_devices} CUDA devices.")

    # Determine which devices to use
    if args.cuda_devices.lower() == "all":
        devices_to_use = list(range(total_devices))
    else:
        try:
            devices_to_use = [int(x.strip()) for x in args.cuda_devices.split(",")]
        except Exception as e:
            raise ValueError(
                "Invalid device IDs provided. Please provide a comma-separated list of integers, e.g., '0,1'."
            ) from e

        # Validate provided device IDs
        for d in devices_to_use:
            if d < 0 or d >= total_devices:
                raise ValueError(
                    f"Device ID {d} is out of range. Available device IDs are 0 to {total_devices - 1}."
                )
    print("Using CUDA devices:", devices_to_use)

    # List all images in the dataset directory
    try:
        all_images = os.listdir(DOP_PATH)
    except FileNotFoundError:
        print(
            f"Directory {DOP_PATH} not found. Please ensure the dataset root contains a 'dop_images' folder."
        )
        return

    if not all_images:
        print("No images found in the dataset directory!")
        return

    # Partition images among selected GPUs (round-robin distribution)
    partitions = {device_id: [] for device_id in devices_to_use}
    num_selected = len(devices_to_use)
    for idx, image in enumerate(all_images):
        device_id = devices_to_use[idx % num_selected]
        partitions[device_id].append(image)

    # Process partitions in parallel using a process pool.
    with ProcessPoolExecutor(max_workers=num_selected) as executor:
        futures = []
        for device_id, image_list in partitions.items():
            if image_list:
                futures.append(
                    executor.submit(
                        process_images_for_device,
                        image_list,
                        device_id,
                        DOP_PATH,
                        EMB_PATH,
                        args.checkpoint,
                    )
                )
        # Wait for all workers to finish
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
