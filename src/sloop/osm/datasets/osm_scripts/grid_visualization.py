import utils
import json
import parse_osm_mapping
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import random
import numpy as np
from time import time
from shapely.geometry import Point, LineString, box
from shapely.geometry.polygon import Polygon
import sys
import pdb

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

CELL_DIAM = 5
MAP_DIAM = 50

idx_to_cell = {}
name_to_idx = {}
idx = 0

# Brown campus
# the point is a center point

### Faunce House ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((41.826771, -71.40255), MAP_DIAM)
# map_data = utils.create_map_from_osm((41.826771, -71.40255), MAP_DIAM)

### CVS on Thayer ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((41.8299353, -71.4007520), MAP_DIAM)
# map_data = utils.create_map_from_osm((41.8299353, -71.4007520), MAP_DIAM)

### The Dorrance ###
nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((41.8244409, -71.4123362), MAP_DIAM)
map_data = utils.create_map_from_osm((41.8244409, -71.4123362), MAP_DIAM)

### New York City ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((40.7736696, -73.9596624), MAP_DIAM)
# map_data = utils.create_map_from_osm((40.7736696, -73.9596624), MAP_DIAM)

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

    for idx in idx_to_cell:
        cell_poly = idx_to_cell[idx].get_poly()
        if cell_poly.intersects(landmark):
            if name_to_idx.get(node_key) == None:
                name_to_idx[node_key] = [idx]
            else:
                name_to_idx[node_key].append(idx)

for key in remove_keys:
    del ways[key]

# minx, miny, maxx, maxy
bbox = box(west_limit, south_limit, east_limit, north_limit)
polyhole = Polygon(bbox.exterior.coords, [w.exterior.coords for w in ways.values()])

# test_pt = Point(41.82388034255718, -71.41229219862235)
# test_pt = Point(41.82379675147419, -71.41240792417489)
pixel_size = 455
counter = 1

def random_points_plot(poly, robot_num, obj_num):
    west, south, east, north = poly.bounds
    width = abs(east - west)
    height = abs(north - south)

    # dictionary of {"0_0": cell_index}
    # where "0_0" represents map name (object and robot counters)
    idx_to_robot_pt = {}
    idx_to_obj_pt = {}

    # dictionary of {"0_0": latitude and longitude tuple},
    # where "0_0" represents map name (object and robot counters)
    robot_coords = {}
    obj_coords = {}

    robot_counter = 0
    # robot_pts = []
    # obj_pts = []
    while robot_counter < robot_num:
        robot_rand_lon = random.uniform(west + 0.0001, east - 0.0001)
        robot_rand_lat = random.uniform(north - 0.0001, south + 0.0001)
        robot_pt = Point(robot_rand_lon, robot_rand_lat)
        # obj_pt = geometry.Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        if robot_pt.within(poly):
            obj_counter = 0
            while obj_counter < obj_num:
                obj_rand_lon = random.uniform(west + 0.0001, east - 0.0001)
                obj_rand_lat = random.uniform(north - 0.0001, south + 0.0001)
                obj_pt = Point(obj_rand_lon, obj_rand_lat)
                if obj_pt.within(poly):
                    im = plt.imread("/home/matthew/spatial_lang/dorrance_feb11_maps/dorrance.png")
                    implot = plt.imshow(im)

                    # lat/lon to pixel conversion
                    obj_lon = ((obj_pt.x - west) / width) * pixel_size
                    obj_lat = (abs(obj_pt.y - north) / height) * pixel_size
                    robot_lon = ((robot_pt.x - west) / width) * pixel_size
                    robot_lat = (abs(robot_pt.y - north) / height) * pixel_size
                    obj_plt = plt.scatter([obj_lon], [obj_lat], s=100, c='g')
                    robot_plt = plt.scatter([robot_lon], [robot_lat], s=100, c='r', marker='^')
                    # plt.savefig("/home/matthew/spatial_lang/spatial-lang/dorrance_feb11_maps/dorrance_feb11_" + str(robot_counter) + "_" + str(obj_counter) + ".png", dpi=600)
                    obj_plt.remove()
                    robot_plt.remove()
                    obj_counter += 1
                    idx_to_robot_pt[str(robot_counter) + "_" + str(obj_counter)] = (robot_pt.y, robot_pt.x)
                    idx_to_obj_pt[str(robot_counter) + "_" + str(obj_counter)] = (obj_pt.y, obj_pt.x)
                    # obj_pts.append((obj_pt.y, obj_pt.x))
            robot_counter += 1
            # robot_pts.append((robot_pt.y, robot_pt.x))
    return idx_to_robot_pt, idx_to_obj_pt, robot_coords, obj_coords

robot_pt_dict, obj_pt_dict, robot_coords, obj_coords = random_points_plot(bbox, 30, 10)

# robot_pts_file = open("/home/matthew/spatial_lang/spatial-lang/dorrance_feb11_maps/robot_cells_dorrance_feb11.txt", "w")
# robot_pts_file.write(str(robot_pt_dict))
# robot_pts_file.close()
#
# obj_pts_file = open("/home/matthew/spatial_lang/spatial-lang/dorrance_feb11_maps/object_cells_dorrance_feb11.txt", "w")
# obj_pts_file.write(str(obj_pt_dict))
# obj_pts_file.close()
#
# robot_coords_file = open("/home/matthew/spatial_lang/spatial-lang/dorrance_feb11_maps/robot_coords_dorrance_feb11.txt", "w")
# robot_coords_file.write(str(robot_coords))
# robot_coords_file.close()
#
# obj_coords_file = open("/home/matthew/spatial_lang/spatial-lang/dorrance_feb11_maps/object_coords_dorrance_feb11.txt", "w")
# obj_coords_file.write(str(obj_coords))
# obj_coords_file.close()
#
# idx_to_cell_json = json.dumps(idx_to_cell, default=jdefault)
# name_to_idx_json = json.dumps(name_to_idx, default=jdefault)
#
# idx_to_cell_file = open("/home/matthew/spatial_lang/spatial-lang/dorrance_feb11_maps/idx_to_cell_dorrance_feb11.json", "w")
# idx_to_cell_file.write(idx_to_cell_json)
# idx_to_cell_file.close()
#
# name_to_idx_file = open("/home/matthew/spatial_lang/spatial-lang/dorrance_feb11_maps/name_to_idx_dorrance_feb11.json", "w")
# name_to_idx_file.write(str(name_to_idx_json))
# name_to_idx_file.close()


### TODO ###
# utf-8 problems
# not handling the possibility of multiple places with same name
# visualization?
