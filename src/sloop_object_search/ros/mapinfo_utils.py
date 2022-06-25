# We don't want to directly modify MapInfoDataset as it
# is a dead class part of a Google-Drive-saved zip file.
#
# Instead we create some functionality related to mapinfo
# here.
import os
from sloop.osm.datasets import FILEPATHS, MapInfoDataset

# Path to the 'for_robots' folder that contains maps for robot tests.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../../data/robot_tests")

def _create_fp_dict_new(map_names):
    """WARNING: for NEW FORMAT (not original OSM)"""
    fp_dict = {}
    for map_name in map_names:
        fp_dict[map_name] = {
            "name_to_symbols": os.path.join(DATA_PATH, map_name, "name_to_symbols.json"),
            "symbol_to_grids": os.path.join(DATA_PATH, map_name, "symbol_to_grids.json"),
            "symbol_to_synonyms": os.path.join(DATA_PATH, map_name, "symbol_to_synonyms.json"),
            "streets": os.path.join(DATA_PATH, map_name, "streets.json"),
            "map_dims": os.path.join(DATA_PATH, map_name, "map_dims.json"),
            "excluded_symbols": os.path.join(DATA_PATH, map_name, "excluded_symbols.json"),
            "map_png": os.path.join(DATA_PATH, map_name, "%s.PNG" % map_name)
        }
    return fp_dict

# We will update FILEPATHS with additional maps in DATA_PATH
FILEPATHS.update(_create_fp_dict_new(os.listdir(DATA_PATH)))


################## Functions ##############################3
def register_map(grid_map):
    """
    Registers a new map; This means creating a folder under DATA_PATH
    that is named 'map_name' and its content structured just like the
    other OSM maps. Of course, we will skip all the geo-info files.
    Those don't really matter; See neighborhoods for an example.

    Note that the grid_map itself is not saved; only the landmarks.
    """
    map_name = grid_map.name
    mapdir = os.path.join(DATA_PATH, map_name)
    if os.path.exists(mapdir):
        raise ValueError(f"map {map_name} is already present.")

    os.mkdir(mapdir)
    with open(os.path.join(mapdir, "excluded_symbols.json"), "w") as f:
        f.write("[]")
    with open(os.path.join(mapdir, "name_to_symbols.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(mapdir, "symbol_to_grids.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(mapdir, "symbol_to_synonyms.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(mapdir, "streets.json"), "w") as f:
        f.write("[]")
    with open(os.path.join(mapdir, "map_dims.json"), "w") as f:
        f.write(f"[{grid_map.width}, {grid_map.length}]")
    print(f"Map {grid_map.name} registered")
    FILEPATHS.update(_create_fp_dict_new([map_name]))
