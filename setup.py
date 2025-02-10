from setuptools import find_packages, setup

setup(
    name="roofsam",
    version="1.0",
    install_requires=[
        "tqdm",
        "shapely>=2.0.7",
        "scikit-learn>=1.6.1",
        "pillow>=11.1.0",
        "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git"
    ],
    packages=find_packages(),
    extras_require={
        # dependencies for roofsam_build_alkis_roof_dataset_wcs.py
        # the recommended way is to use docker instead
        "all": ["pyshp>=2.3.1", "Rtree>=1.3.0", "requests>=2.32.3"],    
    },
    scripts=["tools/build_alkis_dataset/roofsam_build_alkis_roof_dataset_wcs.py"
             "tools/roofsam_precompute_embeddings_cuda.py",
             "tools/roofsam_test.py",
             "tools/roofsam_train.py"]
)
