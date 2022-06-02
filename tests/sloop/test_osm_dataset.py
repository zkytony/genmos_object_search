import cv2
import pytest

from sloop.osm.datasets import MapInfoDataset, FILEPATHS
import subprocess

def test_dataset_mapinfo():
    p = subprocess.Popen(["python", "../../src/sloop/osm/datasets/SL_OSM_Dataset/mapinfo/map_info_dataset.py"])
