
# RoofSAM: Adapting the Segment Anything Model to Rooftop Classification in Aerial Images

**[ASILOMAR Conference on Signals, Systems, and Computers 2024 Paper](#)**

RoofSAM adapts the powerful [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) for the specialized task of classifying rooftop shapes in aerial imagery. By leveraging predefined building footprint polygons, RoofSAM uses point sampling to guide an adapted mask decoder, which then produces a point-wise roof shape class distribution. Final roof classifications are determined by majority voting over the sampled points.

<p align="center">
  <img src="assets/roofsam.png" alt="RoofSAM Architecture" width="700"/>
</p>

The accompanying paper presents experiments on point sampling strategies and investigates the effect of varying the number of sampling points per roof instance.

---

## Table of Contents

- [Installation](#installation)
- [Building the Dataset](#building-the-dataset)
- [Model Checkpoints](#model-checkpoints)
- [Tools](#tools)
- [License](#license)
- [Citing RoofSAM](#citing-roofsam)
- [Acknowledgements](#acknowledgements)

---

## Installation

RoofSAM requires:
- **Python**: version `>=3.10.8`
- **PyTorch**: version `>=2.0.1`
- **TorchVision**: version `>=0.15.2`

*Note:* For optimal performance, install PyTorch and TorchVision with CUDA support. Follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) to set up your environment.

### Installing RoofSAM

You can install RoofSAM directly from GitHub:

```bash
pip install git+https://github.com/tritolol/RoofSAM
```

Or, to install locally:
```bash
git clone git@github.com:tritolol/RoofSAM.git
cd roofsam
pip install -e .
```
To include all optional dependencies (useful if you plan to build the dataset without Docker), run:
```bash
cd roofsam
pip install -e .[all]
```
Note: Ensure that wget, unzip, and ogr2ogr (from GDAL) are installed and available in your system PATH.

## Building the Dataset

The experiments in the paper used a proprietary dataset from Wuppertal, Germany. To facilitate reproducibility, we provide a script that builds a comparable dataset using aerial imagery from the publicly available [North-Rhine Westphalia geo data portal](https://www.opengeodata.nrw.de/produkte/). This dataset offers imagery at resolutions up to 10cm/pixel—matching the quality and resolution used in the paper.

The dataset creation script:
- Downloads building cadastre data.
- Filters for roof categories.
- Queries the [Web Coverage Service (WCS)](https://en.wikipedia.org/wiki/Web_Coverage_Service) for aerial images at specified locations.

Since the script relies on some uncommon system libraries (e.g., GDAL), we provide a Docker image to simplify the setup.

### Building the Dataset with Docker
1. Create a Dataset Directory:
    ```bash
    mkdir dataset
    ```

2. Choose One of the Following Methods:
    #### Method 1: Use the Pre-built Docker Hub Image
    Run the container with the pre-built image tritolol/roofsam-dataset:
    ```bash
    docker run --rm --mount type=bind,src=./dataset,dst=/dataset tritolol/roofsam-dataset /venv/bin/python /app/roofsam_build_alkis_roof_dataset_wcs.py --output-dir /dataset
    ```
    #### Method 2: Build the Docker Image Locally
    Build the Docker image:
    ```bash
    docker build -t dataset_builder tools/build_alkis_dataset
    ```
    ```bash
    docker run --rm --mount type=bind,src=./dataset,dst=/dataset dataset_builder /venv/bin/python /app/roofsam_build_alkis_roof_dataset_wcs.py --output-dir /dataset
    ```

For additional configuration options, you can view the help message:
```bash
docker run --rm dataset_builder /venv/bin/python /app/roofsam_build_alkis_roof_dataset_wcs.py --help
```

## Model Checkpoints
RoofSAM is comprised of two key components:

1. *SAM Image Encoder*:
    The encoder weights can be downloaded from the [SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints). These weights are automatically downloaded when running the precomputation script.
2. *Adapted Mask Decoder*:
    Pre-trained weights for the mask decoder are available in the checkpoints folder. For example, the checkpoint decoder_wuppertal_0.2.pt was trained using data from Wuppertal with a ground sampling resolution of 0.2m/pixel and 4 sampling points.

## Tools
The repository provides several command-line tools located in the `tools/` directory. These scripts are installed to your PATH during setup and can be configured via command-line arguments (use `-h` for help).

- Embedding Precomputation:

    `roofsam_precompute_embeddings_cuda.py`

    Description: Precompute image embeddings using the SAM image encoder across one or multiple CUDA devices. These embeddings are required for both training and testing.
- Training:

    `roofsam_train.py`

    Description: Train the RoofSAM model using the provided dataset. Requires precomputed image embeddings.
- Testing:

    `roofsam_test.py`

    Description: Evaluate a trained model. Also requires precomputed image embeddings.

Usage example to see all available options for a tool:
```bash
roofsam_train.py -h
```

## License

The repository is licensed under the [Apache 2.0 license](LICENSE).

## Citing RoofSAM

```bibtex
@misc{TBA,
  title={RoofSAM: Adapting the Segment Anything Model to Rooftop Classification in Aerial Images},
  note={To appear in ASILOMAR Conference on Signals, Systems, and Computers 2024}
}
```
*Note*: Citation details will be updated upon publication.

## Acknowledgements

<details>
    <summary>
        <a href="https://github.com/facebookresearch/segment-anything">SAM</a> (Segment Anything) [<b>bib</b>]
    </summary>

```bibtex
@article{kirillov2023segany,
title={Segment Anything}, 
author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
journal={arXiv:2304.02643},
year={2023}
}
```
</details>