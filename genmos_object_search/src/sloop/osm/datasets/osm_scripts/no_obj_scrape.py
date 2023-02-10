import utils
import json
from unidecode import unidecode
import parse_osm_mapping
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random
import numpy as np
from time import time
from shapely.geometry import Point, LineString, box
from shapely.geometry.polygon import Polygon
import sys
import os
from city_centers import CITY_CENTERS
import itertools
import copy

OSM_SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))
# CELL_DIAM = 1
CELL_DIAM = 5
MAP_DIAM = 100

class GridCell(object):
    def __init__(self, nw, ne, sw, se):
        """
        ne: tuple representing lat/lon of northeast corner of grid cell
        nw: tuple representing lat/lon of northwest corner of grid cell
        sw: tuple representing lat/lon of southwest corner of grid cell
        se: tuple representing lat/lon of southeast corner of grid cell
        """
        # four corners
        self.nw = nw
        self.ne = ne
        self.sw = sw
        self.se = se

    def toJSON(self):
        return json.dumps(self, default=lambda o: vars(o), sort_keys=True, indent=4)

    # Shapely polygon representation
    def get_poly(self):
        return Polygon([self.nw, self.sw, self.se, self.ne])

        # an approximate center of the cell
        # self.center = ((ne[0] + se[0]) / 2.0, (ne[1] + nw[1]) / 2.0)

    def is_in(self, lat, lon):
        """
        Determine if a lat/lon point is within this cell.
        """
        return lat >= self.se[0] and lat <= self.ne[0] \
            and lon >= self.nw[1] and lon <= self.ne[1]

    """
    def __dict__(self):
        return {"nw": self.nw, "ne": self.ne, "sw": self.sw, "se": self.se}

    def __str__(self):
        return "{nw: " + str(self.nw) + ", ne: " + str(self.ne) + ", sw: ", + str(self.sw) + ", se: ", + str(self.se) + "}"
    """

def jdefault(obj):
    return obj.__dict__

def longest_row(idx_to_cell):
    """
    Identify the number of cells in the longest row of the map.
    A new row starts when the north latitude changes (ratio < 1).
    """
    row_counter = 0
    row_max = 0
    for i in range(len(idx_to_cell) - 1):
        n = idx_to_cell[i].nw[0]
        n1 = idx_to_cell[i+1].nw[0]
        # check if same row
        if (n1 / n) == 1:
            row_counter += 1
        else: # new row
            row_max = max(row_max, row_counter)
            row_counter = 0
    return row_max

def create_pomdp_to_idx(idx_to_cell):
    """
    Creates the pomdp_to_idx file by processing the idx_to_cell dict.
    """
    row_len = longest_row(idx_to_cell)

    pomdp_to_idx = {}
    row_len_counter = 0 # how far into the row
    row_idx_counter = 0 # which row

    pomdp_to_idx[str((row_len_counter, row_idx_counter))] = 0
    row_len_counter += 1
    for i in range(1, len(idx_to_cell)):
        n = idx_to_cell[i-1].nw[0]
        n1 = idx_to_cell[i].nw[0]
        ratio = n1 / n
        if ratio < 1:
            row_len_counter = 0
            row_idx_counter += 1
            # pomdp_to_idx[str((row_len_counter, 0))] = i
        pomdp_to_idx[str((row_len_counter, row_idx_counter))] = i
        row_len_counter += 1

    return pomdp_to_idx

final_replace_dict = {"Street": "St", "Quadrangle":"Quad"}

def format_symbol(name):
    """
    Formats a sentence (iterable of words).
    """
    wrd_lst = unidecode(str(name)).strip() \
        .replace('\n', '') \
        .replace('\r', '') \
        .replace(',','') \
        .replace('.','') \
        .replace('?','') \
        .replace('!','') \
        .replace('&','and') \
        .replace("7-", "Seven") \
        .replace("-", "") \
        .split()

        # .replace("+", "") \

    final_word = wrd_lst[-1]
    if final_word in final_replace_dict:
        wrd_lst[-1] = final_replace_dict[final_word]
    symbol = "".join(wrd_lst)
    return symbol

def create_name_to_symbols(name_to_idx):
    symbol_to_name = {}
    name_to_symbols = {}

    for name in name_to_idx:
        symbol = format_symbol(name)
        # TODO: check for uniqueness?
        name_to_symbols[name] = symbol
        symbol_to_name[symbol] = name

    return symbol_to_name, name_to_symbols

def scrape_osm(city_name):
    idx_to_cell = {}
    name_to_idx = {}
    name_to_feats = {}
    cardinal_to_limit = {}
    idx = 0

    nw_limit, ne_limit, sw_limit, se_limit = utils.bbox(CITY_CENTERS[city_name], MAP_DIAM)
    map_data = utils.create_map_from_osm(CITY_CENTERS[city_name], MAP_DIAM)

    map_data = parse_osm_mapping.parse_json(map_data)
    parse_osm_mapping.build_ways(map_data)
    ways = parse_osm_mapping.get_ways()
    nodes = parse_osm_mapping.get_nodes()
    buildings = parse_osm_mapping.get_buildings()
    building_polys = []

    north_limit = nw_limit[0]
    east_limit = se_limit[1]
    south_limit = se_limit[0]
    west_limit = nw_limit[1]
    cardinal_to_limit = {"N": north_limit, "E": east_limit, "S": south_limit, "W": west_limit}

    print("north limit: ", north_limit)
    print("east limit: ", east_limit)
    print("south limit: ", south_limit)
    print("west limit: ", west_limit)

    curr_nw = nw_limit
    curr_ne = ne_limit
    curr_sw = sw_limit
    curr_se = se_limit

    # build the idx_to_cell dictionary
    # set indices from west to east, north to south
    while curr_nw[0] > south_limit:
        while curr_nw[1] < east_limit:
            nw, ne, sw, se = utils.bbox_cell(curr_nw, CELL_DIAM)
            cell = GridCell(nw, ne, sw, se)
            idx_to_cell[idx] = cell
            idx += 1
            curr_nw = ne
            curr_sw = se
        curr_nw = (curr_sw[0], west_limit)

    for building in buildings:
        building_polys.append(Polygon(building))

    remove_keys = []
    for way_key in ways:
        way_length = len(ways[way_key])
        if (way_length > 2):
            ways[way_key] = Polygon(ways[way_key])
            way_poly = ways[way_key]
            print("way poly is valid: ", way_poly.is_valid)
            for idx in idx_to_cell:
                cell_poly = idx_to_cell[idx].get_poly()
                if cell_poly.intersects(way_poly):
                    if name_to_idx.get(way_key) == None:
                        name_to_idx[way_key] = [idx]
                    else:
                        name_to_idx[way_key].append(idx)
            if isinstance(way_poly, Polygon):
                major_axis, minor_axis = utils.calc_axis(way_poly)
                landmark_feats = [way_poly.area, way_poly.length, major_axis, minor_axis]
                name_to_feats[way_key] = landmark_feats
            else:
                # building.area, building.length, major_axis, minor_axis
                # TODO: check if this is not creating to much noise
                landmark_feats = [0.0, 1.0, [1, 0], [0, 1]]
                name_to_feats[way_key] = landmark_feats
        else:
            remove_keys.append(way_key)

    for node_key in nodes:
        nodes[node_key] = Point(nodes[node_key])
        landmark = nodes[node_key]

        # Check if a named node is in a building
        for building in building_polys:
            if building.intersects(nodes[node_key]):
                # treat the node like the building instead
                landmark = building
                break

        for idx in idx_to_cell:
            cell_poly = idx_to_cell[idx].get_poly()
            if cell_poly.intersects(landmark):
                if name_to_idx.get(node_key) == None:
                    name_to_idx[node_key] = [idx]
                else:
                    name_to_idx[node_key].append(idx)

        # TODO: not sure if this instance thing is working, there are only two landmarks when there should be more
        if isinstance(landmark, Polygon):
            major_axis, minor_axis = utils.calc_axis(landmark)
            landmark_feats = [landmark.area, landmark.length, major_axis, minor_axis]
            name_to_feats[node_key] = landmark_feats
        else:
            # building.area, building.length, major_axis, minor_axis
            # TODO: check if this is not creating to much noise
            landmark_feats = [0.0, 1.0, [1, 0], [0, 1]]
            name_to_feats[node_key] = landmark_feats

    for key in remove_keys:
        del ways[key]

    # minx, miny, maxx, maxy
    bbox = box(west_limit, south_limit, east_limit, north_limit)
    polyhole = Polygon(bbox.exterior.coords, [w.exterior.coords for w in ways.values()])

    return bbox, idx_to_cell, name_to_idx, cardinal_to_limit, name_to_feats

if __name__ == "__main__":
    city_to_limit = {}
    # cities = CITY_CENTERS
    cities = ["honolulu", "denver", "cleveland", "washington_dc", "austin"]

    for city_name in cities:
        print("____________________________________")
        print(city_name)
        FINE_GRAIN_MAP_PATH = OSM_SCRIPTS_PATH + "/../map_info_dicts/jul8/" + city_name
        if not os.path.exists(FINE_GRAIN_MAP_PATH):
            os.makedirs(FINE_GRAIN_MAP_PATH)
        bbox, idx_to_cell, name_to_idx, cardinal_to_limit, name_to_feats = scrape_osm(city_name)
        pomdp_to_idx = create_pomdp_to_idx(idx_to_cell)
        symbol_to_name, name_to_symbols = create_name_to_symbols(name_to_idx)

        idx_to_cell_json = json.dumps(idx_to_cell, default=jdefault)
        name_to_idx_json = json.dumps(name_to_idx, default=jdefault)
        cardinal_to_limit_json = json.dumps(cardinal_to_limit, default=jdefault)
        name_to_feats_json = json.dumps(name_to_feats, default=jdefault)
        pomdp_to_idx_json = json.dumps(pomdp_to_idx, default=jdefault)
        name_to_symbols_json = json.dumps(name_to_symbols, default=jdefault)
        symbol_to_name_json = json.dumps(symbol_to_name, default=jdefault)

        idx_to_cell_file = open(FINE_GRAIN_MAP_PATH + "/idx_to_cell.json", "w")
        idx_to_cell_file.write(idx_to_cell_json)
        idx_to_cell_file.close()

        name_to_idx_file = open(FINE_GRAIN_MAP_PATH + "/name_to_idx.json", "w")
        name_to_idx_file.write(str(name_to_idx_json))
        name_to_idx_file.close()

        cardinal_to_limit_file = open(FINE_GRAIN_MAP_PATH + "/cardinal_to_limit.json", "w")
        cardinal_to_limit_file.write(str(cardinal_to_limit_json))
        cardinal_to_limit_file.close()

        name_to_feats_file = open(FINE_GRAIN_MAP_PATH + "/name_to_feats.json", "w")
        name_to_feats_file.write(str(name_to_feats_json))
        name_to_feats_file.close()

        pomdp_to_idx_file = open(FINE_GRAIN_MAP_PATH + "/pomdp_to_idx.json", "w")
        pomdp_to_idx_file.write(str(pomdp_to_idx_json))
        pomdp_to_idx_file.close()

        name_to_symbols_file = open(FINE_GRAIN_MAP_PATH + "/name_to_symbols.json", "w")
        name_to_symbols_file.write(str(name_to_symbols_json))
        name_to_symbols_file.close()

        symbol_to_name_file = open(FINE_GRAIN_MAP_PATH + "/symbol_to_name.json", "w")
        symbol_to_name_file.write(str(symbol_to_name_json))
        symbol_to_name_file.close()
