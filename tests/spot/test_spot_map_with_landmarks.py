import cv2
import time
import os
import torch
import spacy
import json
import matplotlib.pyplot as plt
import sloop.observation

from sloop_object_search.oopomdp.models.grid_map import GridMap
from sloop_object_search.utils.visual import GridMapVisualizer
from sloop_object_search.ros.mapinfo_utils import (FILEPATHS,
                                                   MapInfoDataset,
                                                   register_map,
                                                   load_filepaths)
from sloop_object_search.utils.colors import random_unique_color, rgb_to_hex

def test():
    map_name = "lab121_lidar"
    grid_size = 0.4

    mapinfo = MapInfoDataset()
    if load_filepaths(map_name, 0.4):
        mapinfo.load_by_name(map_name)
    else:
        raise ValueError(f"{map_name} does not exist.")

    grid_map = GridMap.load(FILEPATHS[map_name]["grid_map"])
    viz = GridMapVisualizer(grid_map=grid_map, res=30)

    landmark_colors = {}
    img = viz.render()
    landmarks = dict(mapinfo.landmarks[map_name])
    landmarks = {"Abducting_Tv"}
    for landmark_symbol in landmarks:
        _colors = set(rgb_to_hex(landmark_colors[s]) for s in landmark_colors)
        if landmark_symbol not in landmark_colors:
            landmark_colors[landmark_symbol] = random_unique_color(_colors, fmt="rgb")
        landmark_footprint = mapinfo.landmarks[map_name][landmark_symbol]
        img = viz.highlight(img, landmark_footprint, landmark_colors[landmark_symbol])
    img = cv2.flip(img, 1)
    viz.show_img(img)
    time.sleep(5)

if __name__ == "__main__":
    test()
