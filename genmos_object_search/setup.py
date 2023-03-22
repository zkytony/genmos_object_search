#!/usr/bin/env

from setuptools import setup, find_packages

setup(name='genmos_object_search',
      packages=find_packages(),#['genmos_object_search', 'sloop'],
      package_dir={'': 'src'},
      version='0.2',
      description="Generalized Multi-Object Search Package",
      install_requires=[
          'pomdp-py~=1.3',
          'numpy>=1.22.1',
          'matplotlib>=3.5.3',
          'pygame>=2.1.2',
          'spacy>=3.3.1',
          'grpcio>=1.47.0',
          'pyyaml',
          'protobuf>=3.20.3',
          'pandas>=1.4.1',
          'open3d>=0.15.2'
      ],
      author='Kaiyu Zheng',
      author_email='kzheng10@cs.brown.edu',
      keywords=["object search", "spatial language", "POMDP", "SLOOP"]
)
