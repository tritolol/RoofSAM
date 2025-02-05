import os
from collections import Counter
import json
import random

import torch
from torch.utils.data import Dataset, random_split

from shapely import Polygon, Point


# Mapping from numerical class code to human-readable class name.
# If a polygon’s class code is not found in this dictionary,
# it will default to "Sonstiges".
# from https://www.adv-online.de/GeoInfoDok/Aktuelle-Anwendungsschemata/AAA-Anwendungsschema-7.1.2-Referenz-7.1/binarywriterservlet?imgUid=0ef70989-a7b6-0581-9393-b216067bef8a&uBasVariant=11111111-1111-1111-1111-111111111111#_C11223-_A11223_47893
CLASS_CODE_MAPPING = {
    1000: "Flachdach",
    2100: "Pultdach",
    2200: "Versetztes Pultdach",
    3100: "Satteldach",
    3200: "Walmdach",
    3300: "Krüppelwalmdach",
    3400: "Mansardendach",
    3500: "Zeltdach",
    3600: "Kegeldach",
    3700: "Kuppeldach",
    3800: "Sheddach",
    3900: "Bogendach",
    4000: "Turmdach",
    5000: "Mischform",
    9999: "Sonstiges"
}

class RoofDataset(Dataset):
    def __init__(self, dataset_root, num_sampled_points=10, min_samples_threshold=10):
        """
        Args:
            dataset_root (str): Root directory of the dataset.
            num_sampled_points (int): Number of points to sample from each polygon.
            min_samples_threshold (int): Minimum number of samples required for a class to be kept.
                                         Classes with fewer samples are mapped to "other".
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.emb_path = os.path.join(self.dataset_root, "dop_embeddings")
        self.image_path = os.path.join(self.dataset_root, "dop_images")
        self.metadata_path = os.path.join(self.dataset_root, "metadata.json")

        self.num_sampled_points = num_sampled_points
        self.min_samples_threshold = min_samples_threshold

        self.consistency_check()

        self.tiles = self.load_metadata()
        self.polygons, self.classes = self.extract_polygons_and_classes()

        # Remap classes with insufficient samples to "other"
        self.apply_threshold_to_classes()

        # Create a mapping from class name to index (sorted for consistent ordering)
        self.class_to_index = {cls: idx for idx, cls in enumerate(sorted(set(self.classes)))}

    def consistency_check(self):
        if not os.path.exists(self.emb_path):
            raise FileNotFoundError(f"Missing embeddings directory: {self.emb_path}")
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Missing images directory: {self.image_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Missing metadata file: {self.metadata_path}")

        embedding_names = set(x.split('.')[0] for x in os.listdir(self.emb_path))
        image_names = set(x.split('.')[0] for x in os.listdir(self.image_path))
        if embedding_names != image_names:
            raise ValueError("Image names and embedding names do not match")

    def load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    def extract_polygons_and_classes(self):
        """
        Iterates through the metadata tiles and extracts polygons with the associated
        embedding key and class name. The class code is mapped to a human-readable
        name using the CLASS_CODE_MAPPING. If the code is unknown, defaults to "Sonstiges".
        """
        polygons = []
        classes = []
        for tile in self.tiles:
            if "contained_polygons" in tile:
                for poly in tile["contained_polygons"]:
                    # Get the numeric class code from the metadata.
                    class_code = poly["class"]
                    # Map the numeric code to the class name; default to "Sonstiges" if missing.
                    class_name = CLASS_CODE_MAPPING.get(class_code, "Sonstiges")

                    polygons.append({
                        "polygon": Polygon(poly["points"]),
                        "embedding_key": tile["dop_name"].split(".")[0],
                        "class": class_name
                    })
                    classes.append(class_name)
        return polygons, classes

    def apply_threshold_to_classes(self):
        """
        Remap any class that does not have at least `min_samples_threshold` samples to "other".
        After remapping, updates the self.classes list accordingly.
        """
        class_counts = Counter(self.classes)
        # Identify classes below the threshold (do not remap if already "other")
        low_freq_classes = {cls for cls, count in class_counts.items() if count < self.min_samples_threshold and cls != "other"}
        if low_freq_classes:
            for poly in self.polygons:
                if poly["class"] in low_freq_classes:
                    poly["class"] = "other"
            # Update the classes list based on the remapped polygons.
            self.classes = [poly["class"] for poly in self.polygons]

    def get_num_classes(self):
        return len(self.class_to_index.keys())

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, idx):
        polygon_data = self.polygons[idx]
        polygon = polygon_data["polygon"]
        embedding_key = polygon_data["embedding_key"]
        # Look up the index from the updated mapping.
        class_label = self.class_to_index[polygon_data["class"]]

        embedding_path = os.path.join(self.emb_path, f"{embedding_key}.pt")
        if not os.path.exists(embedding_path):
            raise ValueError(f"Embedding not found for key {embedding_key}")
        embedding = torch.load(embedding_path, map_location='cpu').squeeze(0)  # remove batch dimension

        # Sample points from within the polygon.
        min_x, min_y, max_x, max_y = polygon.bounds
        sampled_points = []
        while len(sampled_points) < self.num_sampled_points:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(random_point):
                sampled_points.append((random_point.x, random_point.y))

        sampled_points_tensor = torch.tensor(sampled_points, dtype=torch.float32)
        class_tensor = torch.tensor(class_label, dtype=torch.long)
        return embedding, sampled_points_tensor, class_tensor

    @staticmethod
    def get_train_test_split(dataset_root, train_ratio=0.8, seed=42, num_sampled_points=10, min_samples_threshold=10):
        """
        Splits the dataset into training and testing datasets.
        Also computes class weights based on the updated class mapping.

        Args:
            dataset_root (str): Path to the dataset root.
            train_ratio (float): Ratio of the dataset to use for training (default: 0.8).
            seed (int): Random seed for reproducibility.
            num_sampled_points (int): Number of points to sample from each polygon.
            min_samples_threshold (int): Minimum samples required for a class before it is mapped to "other".

        Returns:
            tuple: (train_dataset, test_dataset, num_classes, class_to_index, class_weights)
        """
        # Create the full dataset with the specified threshold.
        full_dataset = RoofDataset(dataset_root, num_sampled_points=num_sampled_points, min_samples_threshold=min_samples_threshold)

        # Updated class mapping and number of classes.
        class_to_index = full_dataset.class_to_index
        num_classes = len(class_to_index)

        # Compute class distribution based on remapped classes.
        class_counts = Counter(full_dataset.classes)
        total_samples = sum(class_counts.values())

        # Compute class weights in the order of sorted class names (matching class_to_index keys).
        class_weights = torch.tensor(
            [total_samples / (num_classes * class_counts[cls]) for cls in sorted(class_to_index.keys())],
            dtype=torch.float32
        )

        # Ensure reproducibility for the split.
        generator = torch.Generator().manual_seed(seed)
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

        return train_dataset, test_dataset, num_classes, class_to_index, class_weights

if __name__ == "__main__":
    dataset = RoofDataset("dataset")
    item = dataset[0]
