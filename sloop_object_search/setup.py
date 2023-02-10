#!/usr/bin/env

from setuptools import setup, find_packages

setup(name='genmos_object_search',
      packages=find_packages(),#['genmos_object_search', 'sloop'],
      package_dir={'': 'src'},
      version='0.2',
      description="SLOOP object search package",
      install_requires=[
          'pomdp-py',
          'numpy',
          'matplotlib',
          'pygame',
          'spacy',
          'grpcio',
          'pyyaml',
          'protobuf',
          'pandas',
          'open3d'
      ],
      author='Kaiyu Zheng',
      author_email='kzheng10@cs.brown.edu',
      keywords=["object search", "spatial language", "POMDP", "SLOOP"]
)
