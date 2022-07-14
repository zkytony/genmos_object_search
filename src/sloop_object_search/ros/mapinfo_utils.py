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

def _create_fp_dict_new(map_names, grid_size):
    """WARNING: for NEW FORMAT (not original OSM)"""
    fp_dict = {}
    for map_name in map_names:
        fp_dict[map_name] = {
            "name_to_symbols": os.path.join(DATA_PATH, map_name, f"g{grid_size}", "name_to_symbols.json"),
            "symbol_to_grids": os.path.join(DATA_PATH, map_name, f"g{grid_size}", "symbol_to_grids.json"),
            "symbol_to_synonyms": os.path.join(DATA_PATH, map_name, f"g{grid_size}", "symbol_to_synonyms.json"),
            "streets": os.path.join(DATA_PATH, map_name, f"g{grid_size}", "streets.json"),
            "map_dims": os.path.join(DATA_PATH, map_name, f"g{grid_size}", "map_dims.json"),
            "excluded_symbols": os.path.join(DATA_PATH, map_name, f"g{grid_size}", "excluded_symbols.json"),
            "grid_map": os.path.join(DATA_PATH, map_name, f"g{grid_size}", "grid_map.json"),
        }
    return fp_dict

################## Functions ##############################3
def register_map(grid_map, exist_ok=False,
                 symbol_to_grids=None,
                 name_to_symbols=None,
                 symbol_to_synonyms=None,
                 save_grid_map=False):
    """
    Registers a new map; This means creating a folder under DATA_PATH
    that is named 'map_name' and its content structured just like the
    other OSM maps. Of course, we will skip all the geo-info files.
    Those don't really matter; See neighborhoods for an example.

    Note that although grid map is saved here, it is not loaded by
    MapInfoDataset - you need to load it separately. Its path is,
    however, stored in FILEPATHS

    By default, if the map already exists, then we won't register.
    However, one could overwrite with additional data by setting
    'exist_ok' to be True.
    """
    map_name = grid_map.name
    mapdir = os.path.join(DATA_PATH, map_name, f"g{grid_map.grid_size}")
    if os.path.exists(mapdir):
        if not exist_ok:
            raise ValueError(f"map {map_name} is already present.")

    os.mkdir(mapdir)
    with open(os.path.join(mapdir, "excluded_symbols.json"), "w") as f:
        f.write("[]")
    with open(os.path.join(mapdir, "name_to_symbols.json"), "w") as f:
        if name_to_symbols is not None:
            f.write(name_to_symbols)
        else:
            f.write("{}")
    with open(os.path.join(mapdir, "symbol_to_grids.json"), "w") as f:
        if symbol_to_grids is not None:
            f.write(symbol_to_grids)
        else:
            f.write("{}")
    with open(os.path.join(mapdir, "symbol_to_synonyms.json"), "w") as f:
        if symbol_to_synonyms is not None:
            f.write(symbol_to_synonyms)
        else:
            f.write("{}")
    with open(os.path.join(mapdir, "streets.json"), "w") as f:
        f.write("[]")
    with open(os.path.join(mapdir, "map_dims.json"), "w") as f:
        f.write(f"[{grid_map.width}, {grid_map.length}]")

    grid_map.save(os.path.join(mapdir, "grid_map.json"))
    print(f"Map {grid_map.name} registered")
    FILEPATHS.update(_create_fp_dict_new([map_name], grid_size=grid_map.grid_size))

def load_filepaths(map_name, grid_size):
    mapdir = os.path.join(DATA_PATH, map_name, f"g{grid_size}")
    if os.path.exists(mapdir):
        FILEPATHS.update(_create_fp_dict_new([map_name], grid_size=grid_size))
        return True
    else:
        return False
