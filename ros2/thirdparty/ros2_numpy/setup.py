#!/usr/bin/env

from setuptools import setup, find_packages

setup(name='ros2_numpy',
      packages=find_packages(),#['genmos_object_search', 'sloop'],
      package_dir={'': 'src'},
      version='noetic-p1',
      description="Tools for converting ROS messages to and from numpy arrays. (version ROS2 Humble)"
)
