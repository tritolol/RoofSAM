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
        "all": ["pyshp>=2.3.1", "Rtree>=1.3.0"],
    },
)
