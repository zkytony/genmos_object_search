# We don't want to directly modify MapInfoDataset as it
# is a dead class part of a Google-Drive-saved zip file.
#
# Instead we create some functionality related to mapinfo
# here.
import os
import json
from sloop.osm.datasets import FILEPATHS, MapInfoDataset

def _create_fp_dict_new(map_names, grid_size, data_dirpath):
    """WARNING: for NEW FORMAT (not original OSM)"""
    fp_dict = {}
    for map_name in map_names:
        fp_dict[map_name] = {
            "name_to_symbols": os.path.join(data_dirpath, map_name, f"g{grid_size:.2f}", "name_to_symbols.json"),
            "symbol_to_grids": os.path.join(data_dirpath, map_name, f"g{grid_size:.2f}", "symbol_to_grids.json"),
            "symbol_to_synonyms": os.path.join(data_dirpath, map_name, f"g{grid_size:.2f}", "symbol_to_synonyms.json"),
            "streets": os.path.join(data_dirpath, map_name, f"g{grid_size:.2f}", "streets.json"),
            "map_dims": os.path.join(data_dirpath, map_name, f"g{grid_size:.2f}", "map_dims.json"),
            "excluded_symbols": os.path.join(data_dirpath, map_name, f"g{grid_size:.2f}", "excluded_symbols.json"),
            "grid_map": os.path.join(data_dirpath, map_name, f"g{grid_size:.2f}", "grid_map.json"),
        }
    return fp_dict

################## Functions ##############################3
def register_map(grid_map, grid_size, data_dirpath,
                 exist_ok=False,
                 symbol_to_grids=None,
                 name_to_symbols=None,
                 symbol_to_synonyms=None,
                 save_grid_map=False):
    """
    Registers a new map; This means creating a folder under 'data_dirpath'
    that is named 'map_name' and its content structured just like the
    other OSM maps. Of course, we will skip all the geo-info files.
    Those don't really matter; See neighborhoods for an example.

    Note that although grid map is saved here, it is not loaded by
    MapInfoDataset - you need to load it separately. Its path is,
    however, stored in FILEPATHS

    By default, if the map already exists, then we won't register.
    However, one could overwrite with additional data by setting
    'exist_ok' to be True.

    Args:
        grid_map (GridMap or GridMap2)
        grid_size (float): The length of a grid side in meters. This
            is only used to organize the data of this map, if the user
            would like to register the same map with different resolutions.
            By default, this is set to "NA" (not applicable).
        data_dirpath (str): The root directory that holds data for different maps.
        Refer to map_info_dataset.py for the remaining arguments.
    """
    map_name = grid_map.name
    sub_name = f"g{grid_size:.2f}"
    mapdir = os.path.join(data_dirpath, map_name, sub_name)
    if os.path.exists(mapdir):
        if not exist_ok:
            raise FileExistsError(f"map {map_name} is already present.")

    os.makedirs(mapdir, exist_ok=exist_ok)
    with open(os.path.join(mapdir, "excluded_symbols.json"), "w") as f:
        f.write("[]")
    with open(os.path.join(mapdir, "name_to_symbols.json"), "w") as f:
        if name_to_symbols is not None:
            json.dump(name_to_symbols, f, indent=4)
        else:
            f.write("{}")
    with open(os.path.join(mapdir, "symbol_to_grids.json"), "w") as f:
        if symbol_to_grids is not None:
            json.dump(symbol_to_grids, f, indent=4)
        else:
            f.write("{}")
    with open(os.path.join(mapdir, "symbol_to_synonyms.json"), "w") as f:
        if symbol_to_synonyms is not None:
            json.dump(symbol_to_synonyms, f, indent=4)
        else:
            f.write("{}")
    with open(os.path.join(mapdir, "streets.json"), "w") as f:
        f.write("[]")
    with open(os.path.join(mapdir, "map_dims.json"), "w") as f:
        f.write(f"[{grid_map.width}, {grid_map.length}]")

    grid_map.save(os.path.join(mapdir, "grid_map.json"))
    print(f"Map {grid_map.name} registered")
    FILEPATHS.update(_create_fp_dict_new([map_name], grid_size, data_dirpath))
    import pdb; pdb.set_trace()

def load_filepaths(map_name, grid_size, data_dirpath):
    mapdir = os.path.join(data_dirpath, map_name, f"g{grid_size:.2f}")
    if os.path.exists(mapdir):
        FILEPATHS.update(_create_fp_dict_new([map_name], grid_size, data_dirpath))
        return True
    else:
        return False
