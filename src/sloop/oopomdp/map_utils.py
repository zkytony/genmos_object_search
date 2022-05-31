import json
import pickle
import ast
import math
import operator

def get_lang_prior(pomdp_to_map_fp, lang_dict_fp):
    """
    Convert the lang prior dictionary to a format usable by pomdp
    """

    pomdp_to_map = {}
    with open(pomdp_to_map_fp, 'r') as fin:
        pomdp_to_map = json.load(fin)

    lang_dict = {}
    with open(lang_dict_fp, 'rb') as fin:
        lang_dict = pickle.load(fin)

    # https://stackoverflow.com/questions/483666/reverse-invert-a-dictionary-mapping
    map_to_pomdp = {v: k for k, v in pomdp_to_map.items()}

    outer_dict = {}
    for objid in lang_dict.keys():
        inner_dict = {}

        for map_idx in map_to_pomdp:
            observation = 1e-6 # default if there was no observation
            pomdp_tup = map_to_pomdp[map_idx]
            pomdp_tup = ast.literal_eval(pomdp_tup)
            if map_idx in lang_dict[objid]: # observed in prior
                observation = lang_dict[objid][map_idx]
            inner_dict[pomdp_tup] = max(observation, 1e-6)

        # normalize values of dictionary -- prob distro sums to 1
        # https://stackoverflow.com/questions/16417916/normalizing-dictionary-values
        factor=1.0 / math.fsum(inner_dict.values())
        for k in inner_dict:
            inner_dict[k] = inner_dict[k]*factor

        outer_dict[objid] = inner_dict

    return outer_dict


def get_center_latlon(cell_idx, pomdp_to_map_fp, idx_to_cell_fp):
    """
    Arguments:
     - cell_idx: Tuple (row, col) representing a POMDP grid cell index
     - idx_to_cell_fp: String filepath to the idx_to_cell JSON for this map

    Returns:
    Tuple (lat, lon) of the center of that cell
    """

    with open(pomdp_to_map_fp, 'r') as fin:
        pomdp_to_map = json.load(fin)

    with open(idx_to_cell_fp, 'r') as fin:
        idx_to_cell = json.load(fin)

    cell_dict = idx_to_cell[str(pomdp_to_map[str(cell_idx)])]

    # TODO: Is the order of lat lon correct?
    center_lat = (cell_dict["nw"][0] + cell_dict["se"][0]) / 2
    center_lon = (cell_dict["nw"][1] + cell_dict["se"][1]) / 2

    return (center_lat, center_lon)

def latlon_to_pomdp_cell(lat, lon, pomdp_to_map_fp, idx_to_cell_fp):
    with open(pomdp_to_map_fp, 'r') as fin:
        pomdp_to_map = json.load(fin)

    with open(idx_to_cell_fp, 'r') as fin:
        idx_to_cell = json.load(fin)

    idx = None
    # import pdb; pdb.set_trace()
    for i in idx_to_cell.keys():
        cell = idx_to_cell[i]
        if cell_contains(cell, lat, lon):
            idx = i

    # print("map indices: ", sorted(pomdp_to_map.values()))
    for pomdp_idx in pomdp_to_map.keys():
        map_idx = pomdp_to_map[pomdp_idx]
        # print("map_idx: ", map_idx)
        if str(map_idx) == idx:
            return pomdp_idx

    # print("No POMDP cell was found !")
    return "None"

def cell_contains(cell_dict, lat, lon):
    south = cell_dict["sw"][1]
    west = cell_dict["sw"][0]
    north = cell_dict["ne"][1]
    east = cell_dict["ne"][0]

    if west <= lat <= east and south <= lon <= north:
        return True
        # cell_to_coord_dict[map_idx] = int(cell_idx)
        # break

    return round(cell_dict["sw"][0], 5) <= lat <= round(cell_dict["ne"][0], 5) \
    and round(cell_dict["sw"][1], 5) <= lon <= round(cell_dict["ne"][1], 5)
