# Dockerfile for tritolol/roofsam-dataset
FROM ubuntu:24.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -y \
    python3-pip python3.12-venv git wget unzip gdal-bin

RUN python3 -m venv venv
RUN source venv/bin/activate
RUN venv/bin/pip install requests pyshp shapely tqdm rtree pillow numpy

COPY build_alkis_roof_dataset_wcs.py /app/
WORKDIR /app