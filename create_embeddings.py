import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

from segment_anything.segment_anything.build_sam import build_sam_vit_h
from segment_anything.segment_anything.predictor import SamPredictor


DATASET_ROOT = "dataset"
DOP_PATH = os.path.join(DATASET_ROOT, "dop_images")
EMB_PATH = os.path.join(DATASET_ROOT, "dop_embeddings")
os.makedirs(EMB_PATH, exist_ok=True)

if not os.path.exists("sam_vit_h_4b8939.pth"):
    print("Downloading SAM ViT H checkpoint...")
    os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

# TODO: Remove MPS
sam = build_sam_vit_h("sam_vit_h_4b8939.pth").to(device="mps")
predictor = SamPredictor(sam)

images = os.listdir(DOP_PATH)

for image in tqdm(images):
    target_path = os.path.join(EMB_PATH, image.split(".")[0] + ".pt")
    if not os.path.exists(target_path):
        with Image.open(os.path.join(DOP_PATH, image)) as im:
            predictor.set_image(np.array(im))
            emb = predictor.get_image_embedding()
            torch.save(emb, target_path)
