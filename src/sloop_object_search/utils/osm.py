from ..oopomdp.models.grid_map import GridMap
from sloop.osm.datasets import MapInfoDataset

def osm_map_to_grid_map(mapinfo, map_name, landmark_as_obstacles=False):
    w, l = mapinfo.map_dims(map_name)
    obstacles = set()
    if landmark_as_obstacles:
        for landmark in mapinfo.landmarks_for(map_name):
            obstacles |= landmark_footprint(landmark, map_name)
    return GridMap(w, l, obstacles, name=map_name)
